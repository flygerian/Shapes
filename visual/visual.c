
#include "common.h"

#include <sys/ioctl.h>
#include <termios.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void die(const char *message) {
  // editorRefreshScreen();
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
