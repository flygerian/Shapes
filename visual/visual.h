#ifndef shapes_visual_h
#define shapes_visual_h

#include "common.h"
#include <stdio.h>
#include <string.h>

typedef struct {
  int x, y;
  int width, height;
  char *text;
} Box;

void EnableRawMode(Context *ctx);
void DisableRawMode(Context *ctx);

void DrawBox(Context *ctx, Box *box) {
  printf("\033[%d;%dH", box->y, box->x); // Move cursor to position

  printf("┌");
  for (int i = 0; i < box->width - 2; i++)
    printf("─");
  printf("┐");


  // Draw middle lines with text
  for (int i = 1; i < box->height - 1; i++) {
    printf("\033[%d;%dH", box->y + i, box->x);
    printf("│");

    if (i == box->height / 2 && box->text) {
      // Center the text vertically
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

  // Draw bottom border
  printf("\033[%d;%dH", box->y + box->height - 1, box->x);
  printf("└");
  for (int i = 0; i < box->width - 2; i++)
    printf("─");
  printf("┘");

  fflush(stdout);
}

void ClearScreen(Context *ctx) {
  printf("\033[2J"); // Clear screen
  printf("\033[H");  // Move cursor to home
}

#endif
