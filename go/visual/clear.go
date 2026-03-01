package visual

import (
	"fmt"
	"io"
)

// ClearScreen clears the terminal and moves cursor to origin.
func ClearScreen(w io.Writer) {
	fmt.Fprint(w, "\033[2J\033[H")
}
