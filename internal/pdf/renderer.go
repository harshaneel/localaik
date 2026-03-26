package pdf

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

type Renderer interface {
	RenderPDF(ctx context.Context, pdf []byte) ([][]byte, error)
}

type RendererFunc func(ctx context.Context, pdf []byte) ([][]byte, error)

func (f RendererFunc) RenderPDF(ctx context.Context, pdf []byte) ([][]byte, error) {
	return f(ctx, pdf)
}

type ExecRenderer struct {
	Binary string
}

func NewExecRenderer(binary string) *ExecRenderer {
	if binary == "" {
		binary = "pdftoppm"
	}
	return &ExecRenderer{Binary: binary}
}

func (r *ExecRenderer) RenderPDF(ctx context.Context, pdf []byte) ([][]byte, error) {
	workDir, err := os.MkdirTemp("", "localaik-pdf-*")
	if err != nil {
		return nil, fmt.Errorf("create temp dir: %w", err)
	}
	defer os.RemoveAll(workDir)

	inputPath := filepath.Join(workDir, "input.pdf")
	outputPrefix := filepath.Join(workDir, "page")

	if err := os.WriteFile(inputPath, pdf, 0o600); err != nil {
		return nil, fmt.Errorf("write temp PDF: %w", err)
	}

	cmd := exec.CommandContext(ctx, r.Binary, "-png", inputPath, outputPrefix)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("run pdftoppm: %w: %s", err, strings.TrimSpace(string(output)))
	}

	files, err := filepath.Glob(outputPrefix + "-*.png")
	if err != nil {
		return nil, fmt.Errorf("collect rendered pages: %w", err)
	}
	if len(files) == 0 {
		return nil, fmt.Errorf("pdftoppm produced no pages")
	}

	sort.Slice(files, func(i, j int) bool {
		return pageNumberFromFilename(files[i]) < pageNumberFromFilename(files[j])
	})

	rendered := make([][]byte, 0, len(files))
	for _, file := range files {
		page, err := os.ReadFile(file)
		if err != nil {
			return nil, fmt.Errorf("read rendered page %s: %w", file, err)
		}
		rendered = append(rendered, page)
	}

	return rendered, nil
}

func pageNumberFromFilename(file string) int {
	base := strings.TrimSuffix(filepath.Base(file), filepath.Ext(file))
	lastDash := strings.LastIndex(base, "-")
	if lastDash < 0 {
		return 0
	}
	number, err := strconv.Atoi(base[lastDash+1:])
	if err != nil {
		return 0
	}
	return number
}
