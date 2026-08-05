package entrypoint

import (
	"bytes"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"
)

// These drive the real entrypoint.sh with stub children, so no container, model
// or inference engine is involved.

const stubLlamaServer = `#!/bin/sh
echo $$ > "${LK_TEST_LLAMA_PIDFILE}"
if [ -n "${LK_TEST_LLAMA_EXIT_AFTER:-}" ]; then
  sleep "${LK_TEST_LLAMA_EXIT_AFTER}"
  exit "${LK_TEST_LLAMA_STATUS:-0}"
fi
while true; do sleep 1; done
`

const stubProxy = `#!/bin/sh
echo $$ > "${LK_TEST_PROXY_PIDFILE}"
if [ -n "${LK_TEST_PROXY_EXIT_AFTER:-}" ]; then
  sleep "${LK_TEST_PROXY_EXIT_AFTER}"
  exit "${LK_TEST_PROXY_STATUS:-0}"
fi
while true; do sleep 1; done
`

// The line entrypoint.sh prints once its signal handler is installed.
const supervisingMarker = "localaik: supervising"

type syncBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *syncBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (b *syncBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}

type entrypointRun struct {
	cmd          *exec.Cmd
	output       *syncBuffer
	done         chan error
	exited       sync.Once
	exitErr      error
	llamaPIDFile string
	proxyPIDFile string
}

func startEntrypoint(t *testing.T, env ...string) *entrypointRun {
	t.Helper()

	dir := t.TempDir()
	llamaBin := filepath.Join(dir, "llama-server")
	writeStub(t, llamaBin, stubLlamaServer)
	writeStub(t, filepath.Join(dir, "localaik"), stubProxy)

	run := &entrypointRun{
		output:       &syncBuffer{},
		done:         make(chan error, 1),
		llamaPIDFile: filepath.Join(dir, "llama.pid"),
		proxyPIDFile: filepath.Join(dir, "proxy.pid"),
	}

	script, err := filepath.Abs("../../entrypoint.sh")
	if err != nil {
		t.Fatalf("resolve entrypoint.sh: %v", err)
	}

	run.cmd = exec.Command(shellPath(), script)
	run.cmd.Env = append(os.Environ(),
		"LLAMA_SERVER_BIN="+llamaBin,
		"PATH="+dir+string(os.PathListSeparator)+os.Getenv("PATH"),
		"LK_TEST_LLAMA_PIDFILE="+run.llamaPIDFile,
		"LK_TEST_PROXY_PIDFILE="+run.proxyPIDFile,
	)
	run.cmd.Env = append(run.cmd.Env, env...)
	run.cmd.Stdout = run.output
	run.cmd.Stderr = run.output
	// Own process group, so cleanup cannot leave stub children behind.
	run.cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

	if err := run.cmd.Start(); err != nil {
		t.Fatalf("start entrypoint.sh: %v", err)
	}

	go func() { run.done <- run.cmd.Wait() }()

	t.Cleanup(func() {
		_ = syscall.Kill(-run.cmd.Process.Pid, syscall.SIGKILL)
		run.reap()
	})

	return run
}

// The container runs dash, so prefer it when present and fall back to /bin/sh
// rather than silently testing only bash on a developer machine.
func shellPath() string {
	if _, err := os.Stat("/bin/dash"); err == nil {
		return "/bin/dash"
	}
	return "/bin/sh"
}

func writeStub(t *testing.T, path, body string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(body), 0o755); err != nil {
		t.Fatalf("write stub %s: %v", path, err)
	}
}

func (r *entrypointRun) waitForChildren(t *testing.T) (llamaPID, proxyPID int) {
	t.Helper()
	return r.waitForPID(t, r.llamaPIDFile), r.waitForPID(t, r.proxyPIDFile)
}

func (r *entrypointRun) waitForPID(t *testing.T, path string) int {
	t.Helper()

	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		raw, err := os.ReadFile(path)
		if err == nil {
			pid, convErr := strconv.Atoi(strings.TrimSpace(string(raw)))
			if convErr == nil && pid > 0 {
				return pid
			}
		}
		time.Sleep(20 * time.Millisecond)
	}

	t.Fatalf("%s never appeared, so the child was not started.\noutput:\n%s", filepath.Base(path), r.output)
	return 0
}

// Signalling before this line is printed would race the trap installation.
func (r *entrypointRun) waitForSupervising(t *testing.T) {
	t.Helper()

	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if strings.Contains(r.output.String(), supervisingMarker) {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}

	t.Fatalf("entrypoint.sh never reported that it was supervising.\noutput:\n%s", r.output)
}

// reap is idempotent, so a test may wait for the exit itself and still have
// cleanup run without blocking on an already-drained channel.
func (r *entrypointRun) reap() error {
	r.exited.Do(func() { r.exitErr = <-r.done })
	return r.exitErr
}

func (r *entrypointRun) waitForExit(t *testing.T) error {
	t.Helper()

	result := make(chan error, 1)
	go func() { result <- r.reap() }()

	select {
	case err := <-result:
		return err
	case <-time.After(20 * time.Second):
		t.Fatalf("entrypoint.sh never exited.\noutput:\n%s", r.output)
		return nil
	}
}

func (r *entrypointRun) exitStatus(t *testing.T, err error) int {
	t.Helper()

	if err == nil {
		return 0
	}
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) {
		t.Fatalf("entrypoint.sh failed in an unexpected way: %v\noutput:\n%s", err, r.output)
	}
	return exitErr.ExitCode()
}

func (r *entrypointRun) assertReaped(t *testing.T, pid int, name string) {
	t.Helper()

	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if syscall.Kill(pid, 0) != nil {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("%s (pid %d) was still running after entrypoint.sh exited.\noutput:\n%s", name, pid, r.output)
}

// The proxy must listen while the model loads. Gating it behind a readiness
// probe is what once wedged the container instead of failing it.
func TestEntrypointStartsProxyWithoutWaitingForTheModel(t *testing.T) {
	run := startEntrypoint(t)
	llamaPID, proxyPID := run.waitForChildren(t)

	if llamaPID == proxyPID {
		t.Fatalf("expected two distinct children, both were %d", llamaPID)
	}

	exited := make(chan error, 1)
	go func() { exited <- run.reap() }()

	select {
	case err := <-exited:
		t.Fatalf("entrypoint.sh exited early with %v, want it supervising.\noutput:\n%s", err, run.output)
	case <-time.After(2 * time.Second):
	}
}

func TestEntrypointExitsWhenLlamaServerDies(t *testing.T) {
	run := startEntrypoint(t,
		"LK_TEST_LLAMA_EXIT_AFTER=1",
		"LK_TEST_LLAMA_STATUS=3",
	)
	_, proxyPID := run.waitForChildren(t)

	if got := run.exitStatus(t, run.waitForExit(t)); got != 3 {
		t.Fatalf("exit status = %d, want 3 propagated from llama-server", got)
	}
	run.assertReaped(t, proxyPID, "the proxy")
}

func TestEntrypointExitsNonZeroWhenLlamaServerExitsCleanly(t *testing.T) {
	run := startEntrypoint(t,
		"LK_TEST_LLAMA_EXIT_AFTER=1",
		"LK_TEST_LLAMA_STATUS=0",
	)
	run.waitForChildren(t)

	if got := run.exitStatus(t, run.waitForExit(t)); got == 0 {
		t.Fatal("exit status = 0, want non-zero; a container whose engine vanished must not look healthy")
	}
}

func TestEntrypointExitsWhenProxyDies(t *testing.T) {
	run := startEntrypoint(t,
		"LK_TEST_PROXY_EXIT_AFTER=1",
		"LK_TEST_PROXY_STATUS=4",
	)
	llamaPID, _ := run.waitForChildren(t)

	if got := run.exitStatus(t, run.waitForExit(t)); got != 4 {
		t.Fatalf("exit status = %d, want 4 propagated from the proxy", got)
	}
	run.assertReaped(t, llamaPID, "llama-server")
}

func TestEntrypointStopsBothChildrenOnSIGTERM(t *testing.T) {
	run := startEntrypoint(t)
	llamaPID, proxyPID := run.waitForChildren(t)
	run.waitForSupervising(t)

	if err := run.cmd.Process.Signal(syscall.SIGTERM); err != nil {
		t.Fatalf("signal entrypoint.sh: %v", err)
	}

	if got := run.exitStatus(t, run.waitForExit(t)); got != 0 {
		t.Fatalf("exit status = %d, want 0 for a requested stop", got)
	}
	run.assertReaped(t, llamaPID, "llama-server")
	run.assertReaped(t, proxyPID, "the proxy")
}
