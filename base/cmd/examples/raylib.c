#include "raylib.h"
#include <stdio.h>

typedef struct Ball {
  Vector2 coords;
  int radius, xSpeed, ySpeed;
} Ball;

typedef struct Padel {
  Vector2 coords;
  int width, height;
} Padel;

Ball MakeBall(int radius) {
  return (Ball){.radius = radius};
}

void UpdateBallPos(Ball *ball, int screenWidth, int screenHeight) {
  if (ball->coords.x >= screenWidth - 10 || ball->coords.x <= 10) {
    ball->xSpeed *= -1;
  }

  if (ball->coords.y >= screenHeight - 10 || ball->coords.y <= 10) {
    ball->ySpeed *= -1;
  }

  ball->coords.x += ball->xSpeed;
  ball->coords.y += ball->ySpeed;
}

Padel MakePadel(int width, int height) {
  return (Padel){.width = width, .height = height};
}

void CheckPadelCollidesWithBall(Padel player, Ball *ball) {
  bool isColliding = CheckCollisionCircleRec(ball->coords, ball->radius, (Rectangle){.x = player.coords.x, .y = player.coords.y, .width = player.width, .height = player.height});
  if (isColliding) {
    printf("collides");
    ball->xSpeed *= -1;
  }
}

void DrawBall(Ball ball) {
  DrawCircle(ball.coords.x, ball.coords.y, ball.radius, WHITE);
}

void DrawPadel(Padel padel) {
  DrawRectangle(padel.coords.x, padel.coords.y, padel.width, padel.height, WHITE);
}

void cpuControlPlayer(Padel *padel, Ball *ball) {
  float padelCenter = padel->coords.y + (padel->height / 2);

  if (ball->coords.y > padelCenter) {
    padel->coords.y += ball->coords.y - padelCenter;
  } else if (ball->coords.y < padelCenter) {
    padel->coords.y -= ball->coords.y;
  }
}

void raylib_test() {
  int ballX = 400;
  int ballY = 400;
  int screenWidth = 1280;
  int screenHeight = 800;
  int movementSpeed = 5;

  Color green = {20, 160, 133, 255};
  Ball ball = MakeBall(20);
  ball.xSpeed = 3;
  ball.ySpeed = 3;
  ball.coords.x = screenWidth / 2;
  ball.coords.y = screenHeight / 2;

  Padel player1 = MakePadel(25, 120);
  Padel player2 = MakePadel(25, 120);

  player2.coords.x = screenWidth - 25 - 10;
  player2.coords.y = screenHeight / 2 - 60;

  player1.coords.x = 10;
  player1.coords.y = screenHeight / 2 - 60;

  InitWindow(screenWidth, screenHeight, "Raylib test");
  SetTargetFPS(120);

  while (!WindowShouldClose()) {
    BeginDrawing();

    if (IsKeyDown(KEY_UP)) {
      player2.coords.y -= movementSpeed;
    } else if (IsKeyDown(KEY_DOWN)) {
      player2.coords.y += movementSpeed;
    }

    cpuControlPlayer(&player1, &ball);

    DrawLine(screenWidth / 2, 0, screenWidth / 2, screenHeight, WHITE);
    UpdateBallPos(&ball, screenWidth, screenHeight);
    DrawBall(ball);
    DrawPadel(player1);
    DrawPadel(player2);

    CheckPadelCollidesWithBall(player1, &ball);
    CheckPadelCollidesWithBall(player2, &ball);
    ClearBackground(green);
    EndDrawing();
  }

  CloseWindow();
}
