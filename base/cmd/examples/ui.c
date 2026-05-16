#include "raylib/src/raylib.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "tensor/types.h"
#include "time.h"
#include <sys/ioctl.h>
#include "unistd.h"
#include "utils_lib/memory.h"
#define CLAY_IMPLEMENTATION
#include "../../clay/clay.h"

const Clay_Color COLOR_LIGHT = (Clay_Color){224, 215, 210, 255};
const Clay_Color COLOR_RED = (Clay_Color){168, 66, 28, 255};
const Clay_Color COLOR_ORANGE = (Clay_Color){225, 138, 50, 255};
const int FONT_ID_BODY_16 = 0;

void HandleClayErrors(Clay_ErrorData errorData) {
  printf("%s", errorData.errorText.chars);
}

Clay_ElementDeclaration sidebarItemConfig =
    (Clay_ElementDeclaration){.layout = {.sizing = {.width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_FIXED(50)}}, .backgroundColor = COLOR_ORANGE};

// Re-useable components are just normal functions
void SidebarItemComponent() {
  CLAY(CLAY_ID(""), sidebarItemConfig) {
    // children go here...
  }
}

// void drawUI(Memory *memory) {
//   uint64_t totalMemorySize = Clay_MinMemorySize();
//   Clay_Arena arena = Clay_CreateArenaWithCapacityAndMemory(totalMemorySize, memory);
//
//   const Clay_Dimensions screenDimensions = (Clay_Dimensions) {.width = GetScreenWidth() * 0.3,
//   .height = GetScreenHeight()}; clock_t lastRenderTime = 0;
//
//   Clay_Initialize(arena, screenDimensions, ( Clay_ErrorHandler ) {  HandleClayErrors });
//   Font fonts[1];
//   fonts[FONT_ID_BODY_16] = LoadFontEx("resources/Roboto-Regular.ttf", 48, 0, 400);
//   SetTextureFilter(fonts[FONT_ID_BODY_16].texture, TEXTURE_FILTER_BILINEAR);
//   Clay_SetMeasureTextFunction(Raylib_MeasureText, fonts);
//
//   while (!WindowShouldClose()) {
//     Clay_SetLayoutDimensions(screenDimensions);
//     Clay_BeginLayout();
//
//     CLAY(CLAY_ID("OuterContainer"), {
//         .layout = {
//           .sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) },
//           .padding = CLAY_PADDING_ALL(16),
//           .childGap = 16
//         },
//         .backgroundColor = {250,250,255,255}
//     }) {
//       CLAY(CLAY_ID("SideBar"), {
//           .layout = {
//             .layoutDirection = CLAY_TOP_TO_BOTTOM,
//             .sizing = { .width = CLAY__SIZING_TYPE_FIT, .height = CLAY__SIZING_TYPE_FIT },
//             .padding = CLAY_PADDING_ALL(16), .childGap = 16
//           },
//           .backgroundColor = COLOR_LIGHT
//       }) {
//             CLAY(CLAY_ID("ProfilePictureOuter"), {
//                 .layout = {
//                   .sizing = { .width = CLAY_SIZING_GROW(0) },
//                   .padding = CLAY_PADDING_ALL(16),
//                   .childGap = 16,
//                   .childAlignment = { .y = CLAY_ALIGN_Y_CENTER }
//                 },
//                 .backgroundColor = COLOR_RED
//             }) {
//                 CLAY(CLAY_ID("ProfilePicture"), {
//                     .layout = {
//                       .sizing = { .width = CLAY_SIZING_FIXED(60), .height = CLAY_SIZING_FIXED(60)
//                       }
//                     },
//                       // .image = { .imageData = &profilePicture }
//                 }) {}
//                 CLAY_TEXT(CLAY_STRING("Clay - UI Library"), { .fontSize = 24, .textColor = {255,
//                 255, 255, 255} });
//             }
//
//             // Standard C code like loops etc work inside components
//             for (int i = 0; i < 5; i++) {
//                 SidebarItemComponent();
//             }
//
//             CLAY(CLAY_ID("MainContent"), {
//                 .layout = {
//                   .sizing = { .width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_GROW(0) }
//                 },
//                 .backgroundColor = COLOR_LIGHT
//             }) {}
//         }
//     }
//
//     int usedata[1] = { 1};
//
//     Clay_RenderCommandArray renderCommands = Clay_EndLayout((clock() - lastRenderTime));
//
//     BeginDrawing();
//     ClearBackground(BLACK);
//     Clay_Raylib_Render(renderCommands, fonts);
//     EndDrawing();
//
//     lastRenderTime = clock();
//   }
// }

Texture2D getTexture(f32 *nhwc, int H, int W, int C) {
  // convert float [0,1] -> uint8 RGBA expected by raylib
  byte *pixels = malloc(H * W * 4);
  for (int i = 0; i < H * W; i++) {
    pixels[i * 4 + 0] = (byte)(nhwc[i * C + 0] * 255.0f); // R
    pixels[i * 4 + 1] = (byte)(nhwc[i * C + 1] * 255.0f); // G
    pixels[i * 4 + 2] = (byte)(nhwc[i * C + 2] * 255.0f); // B
    pixels[i * 4 + 3] = 255;                              // A
  }

  Image img = {
      .data = pixels,
      .width = W,
      .height = H,
      .mipmaps = 1,
      .format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8,
  };

  Texture2D tex = LoadTextureFromImage(img);
  free(pixels);

  // draw it
  //
  return tex;
}

void basicRaylibWindow(Tensor *imageTensor) {
  const int screenWidth = 800;
  const int screenHeight = 450;

  InitWindow(screenWidth, screenHeight, "sample");

  Texture2D tex = getTexture(imageTensor->values, imageTensor->shape.dims[0], imageTensor->shape.dims[1], imageTensor->shape.dims[2]);

  while (!WindowShouldClose()) {
    BeginDrawing();
    ClearBackground(RAYWHITE);
    // DrawText("Congrats! You created your first window!", 190, 200, 20, LIGHTGRAY);
    DrawTexture(tex, 200, 200, WHITE);
    EndDrawing();
  }

  UnloadTexture(tex);
  CloseWindow();
}
