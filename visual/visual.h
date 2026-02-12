#ifndef shapes_visual_h
#define shapes_visual_h

#include "common.h"

typedef struct {
  int x, y;
  int width, height;
  char *topText;    /* Top line: label | value | grad */
  char *bottomText; /* Bottom line: (operation) */
} Box;

void EnableRawMode(Context *ctx);
void DisableRawMode(Context *ctx);
void DrawBox(Context *ctx, Box *box);
void DrawLine(Context *ctx, Box *from, Box *to);
void ClearScreen(Context *ctx);
void VisualizeOps(Context *ctx, Tensor *t);

#endif
