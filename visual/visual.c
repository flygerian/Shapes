
#include "visual.h"
#include "../tensor/tensor.h"
#include "common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

#define VIS_BOX_WIDTH  20
#define VIS_BOX_HEIGHT 5
#define VIS_V_SPACING  5
#define VIS_MAX_NODES  64

void die(const char *message) {
  perror(message);
  exit(1);
}

void DisableRawMode(Context *ctx) {
  int reponse = tcsetattr(STDIN_FILENO, TCSAFLUSH, ctx->screenConfig->orig_termios);
  if (reponse == -1) {
    die("tcsetattr");
  }
}

void EnableRawMode(Context *ctx) {

  if (tcgetattr(STDIN_FILENO, ctx->screenConfig->orig_termios) == -1) {
    die("tcgetattr");
  }

  struct termios raw = *ctx->screenConfig->orig_termios;
  raw.c_iflag &= ~(BRKINT | ICRNL | INPCK | ISTRIP | IXON);
  raw.c_oflag &= ~(OPOST);
  raw.c_cflag |= (CS8);
  raw.c_lflag &= ~(ECHO | ICANON | IEXTEN | ISIG);
  raw.c_cc[VMIN] = 0;
  raw.c_cc[VTIME] = 1;

  if (tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw) == -1) {
    die("tcsetattr");
  }
}

void DrawLine(Context *ctx, Box *from, Box *to) {
  /* Connect bottom-center of 'from' to top-center of 'to' */
  int startX = from->x + from->width / 2;
  int startY = from->y + from->height;

  int endX = to->x + to->width / 2;
  int endY = to->y - 1;

  /* Vertical segment down from source */
  int midY = (startY + endY) / 2;

  for (int y = startY; y <= midY; y++) {
    printf("\033[%d;%dH│", y, startX);
  }

  /* Horizontal segment from startX to endX at midY */
  if (startX < endX) {
    printf("\033[%d;%dH└", midY, startX);
    for (int x = startX + 1; x < endX; x++) {
      printf("\033[%d;%dH─", midY, x);
    }
    printf("\033[%d;%dH┐", midY, endX);
  } else if (startX > endX) {
    printf("\033[%d;%dH┘", midY, startX);
    for (int x = endX + 1; x < startX; x++) {
      printf("\033[%d;%dH─", midY, x);
    }
    printf("\033[%d;%dH┌", midY, endX);
  }
  /* If startX == endX, the vertical lines are already aligned, no horizontal needed */

  /* Vertical segment down to destination */
  for (int y = midY + 1; y <= endY; y++) {
    printf("\033[%d;%dH│", y, endX);
  }

  fflush(stdout);
}

void DrawBox(Context *ctx, Box *box) {
  printf("\033[%d;%dH", box->y, box->x);

  printf("┌");
  for (int i = 0; i < box->width - 2; i++)
    printf("─");
  printf("┐");

  /* Draw middle lines with text */
  for (int i = 1; i < box->height - 1; i++) {
    printf("\033[%d;%dH", box->y + i, box->x);
    printf("│");

    if (i == box->height / 2 && box->text) {
      int text_len = strlen(box->text);
      int padding = (box->width - 2 - text_len) / 2;
      for (int j = 0; j < padding; j++)
        printf(" ");
      printf("%.*s", box->width - 2 - padding, box->text);
      for (int j = padding + text_len; j < box->width - 2; j++)
        printf(" ");
    } else {
      for (int j = 0; j < box->width - 2; j++)
        printf(" ");
    }

    printf("│");
  }

  /* Draw bottom border */
  printf("\033[%d;%dH", box->y + box->height - 1, box->x);
  printf("└");
  for (int i = 0; i < box->width - 2; i++)
    printf("─");
  printf("┘");

  fflush(stdout);
}

void ClearScreen(Context *ctx) {
  printf("\033[2J");
  printf("\033[H");
}

static const char *opName(OpType op) {
  switch (op) {
    case OP_ADD: return "+";
    case OP_SUBTRACT: return "-";
    case OP_MULTIPLY: return "*";
    case OP_DIVIDE: return "/";
    default: return "?";
  }
}

static char *tensorLabel(Context *ctx, Tensor *t) {
  char *val = GetItem(ctx, t);
  if (t->computation == NULL) {
    return val;
  }

  /* Format: "value (op)" e.g. "-6 (*)" */
  const char *op = opName(t->computation->optype);

  /* Check if grad is available and include it */
  if (t->computation->grad != NULL) {
    char *gradVal = GetItem(ctx, t->computation->grad);
    size_t len = strlen(val) + 6 + strlen(op) + strlen(gradVal);
    char *label = allocate(ctx->memory, len + 1);
    snprintf(label, len + 1, "%s (%s) [%s]", val, op, gradVal);
    return label;
  }

  size_t len = strlen(val) + 4 + strlen(op);
  char *label = allocate(ctx->memory, len + 1);
  snprintf(label, len + 1, "%s (%s)", val, op);
  return label;
}

static char *tensorDisplayLabel(Context *ctx, Tensor *t) {
  char *baseLabel = tensorLabel(ctx, t);
  if (t->label == NULL) {
    return baseLabel;
  }

  /* Format: "Label: value (op)" or "Label: value" */
  size_t len = strlen(t->label) + 2 + strlen(baseLabel);
  char *display = allocate(ctx->memory, len + 1);
  snprintf(display, len + 1, "%s: %s", t->label, baseLabel);
  return display;
}

void VisualizeOps(Context *ctx, Tensor *t) {
  /*
   * BFS traversal of the computation graph.
   * queue[] holds tensor pointers; level[] tracks each node's depth.
   * parentIdx[] records which queue index is the parent of each node
   * so we can draw lines after all boxes are placed.
   */
  Tensor *queue[VIS_MAX_NODES];
  int level[VIS_MAX_NODES];
  int parentIdx[VIS_MAX_NODES];
  int count = 0;

  /* Seed the queue with the root tensor */
  queue[count] = t;
  level[count] = 0;
  parentIdx[count] = -1;
  count++;

  /* BFS expand */
  for (int i = 0; i < count && count < VIS_MAX_NODES; i++) {
    Tensor *cur = queue[i];
    if (cur->computation == NULL) {
      continue;
    }
    for (u8 j = 0; j < cur->computation->numInputs; j++) {
      if (count >= VIS_MAX_NODES) {
        break;
      }
      queue[count] = cur->computation->inputs[j];
      level[count] = level[i] + 1;
      parentIdx[count] = i;
      count++;
    }
  }

  /* Determine how many nodes are at each level */
  int maxLevel = 0;
  for (int i = 0; i < count; i++) {
    if (level[i] > maxLevel) {
      maxLevel = level[i];
    }
  }

  /* Assign box positions: spread nodes at each level evenly across the width */
  Box boxes[VIS_MAX_NODES];
  int levelIdx[VIS_MAX_NODES]; /* per-level running index */
  int levelCount[VIS_MAX_NODES];

  memset(levelCount, 0, sizeof(levelCount));
  memset(levelIdx, 0, sizeof(levelIdx));

  for (int i = 0; i < count; i++) {
    levelCount[level[i]]++;
  }

  for (int i = 0; i < count; i++) {
    int lv = level[i];
    int nodesAtLevel = levelCount[lv];
    int idx = levelIdx[lv]++;

    int totalWidth = nodesAtLevel * VIS_BOX_WIDTH + (nodesAtLevel - 1) * 4;
    int startX = (80 - totalWidth) / 2;
    if (startX < 1) {
      startX = 1;
    }

    boxes[i].x = startX + idx * (VIS_BOX_WIDTH + 4);
    boxes[i].y = 2 + lv * (VIS_BOX_HEIGHT + VIS_V_SPACING);
    boxes[i].width = VIS_BOX_WIDTH;
    boxes[i].height = VIS_BOX_HEIGHT;
    boxes[i].text = tensorDisplayLabel(ctx, queue[i]);
  }

  ClearScreen(ctx);

  /* Draw all boxes */
  for (int i = 0; i < count; i++) {
    DrawBox(ctx, &boxes[i]);
  }

  /* Draw lines from parent to child */
  for (int i = 0; i < count; i++) {
    if (parentIdx[i] >= 0) {
      DrawLine(ctx, &boxes[parentIdx[i]], &boxes[i]);
    }
  }

  /* Move cursor below the drawing */
  int bottomY = 2 + (maxLevel + 1) * (VIS_BOX_HEIGHT + VIS_V_SPACING) + 1;
  printf("\033[%d;%dH", bottomY, 0);
  fflush(stdout);
}
