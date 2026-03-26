package pdf

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestExecRenderer(t *testing.T) {
	dir := t.TempDir()
	script := filepath.Join(dir, "fake-pdftoppm.sh")
	content := "#!/bin/sh\nset -eu\nprefix=\"$3\"\nprintf 'page-two' > \"${prefix}-2.png\"\nprintf 'page-one' > \"${prefix}-1.png\"\n"
	if err := os.WriteFile(script, []byte(content), 0o755); err != nil {
		t.Fatalf("write script: %v", err)
	}

	renderer := NewExecRenderer(script)
	pages, err := renderer.RenderPDF(context.Background(), []byte("%PDF-1.4"))
	if err != nil {
		t.Fatalf("RenderPDF returned error: %v", err)
	}
	if len(pages) != 2 {
		t.Fatalf("page count = %d, want 2", len(pages))
	}
	if string(pages[0]) != "page-one" || string(pages[1]) != "page-two" {
		t.Fatalf("pages = %q, %q; want page-one, page-two", string(pages[0]), string(pages[1]))
	}
}
