package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/schollz/progressbar/v3"
)

func cmdTest(mutDir, relDir string, args []string) error {
	f := flag.NewFlagSet("muzoo test", flag.ContinueOnError)
	jobs := f.Int("j", runtime.NumCPU(), "number of parallel jobs")
	timeout := f.Duration("timeout", 0, "timeout per test invocation")
	memory := f.String("memory", "", "memory limit per test invocation (e.g. 2GiB); mutations exceeding it are killed")
	verbose := f.Bool("v", false, "show output for killed mutations")
	if err := f.Parse(args); err != nil {
		return err
	}
	testCmd := f.Args()

	memLimit, err := parseSize(*memory)
	if err != nil {
		return fmt.Errorf("invalid -memory value: %w", err)
	}

	defaultGoTest := len(testCmd) == 0
	if defaultGoTest {
		testCmd = []string{"go test -json -failfast -parallel 2 -short ./... && go test -json -failfast -parallel 2 ./..."}
	}

	pytestCmd := !defaultGoTest && isPytestCmd(testCmd)
	if pytestCmd {
		testCmd = addPytestFlags(testCmd)
	}

	// List and validate all patches.
	patches, err := listPatches(mutDir)
	if err != nil {
		return fmt.Errorf("listing patches: %w", err)
	}
	if len(patches) == 0 {
		fmt.Println("No mutations found.")
		return nil
	}

	// Warn if there are uncommitted changes outside the mutations directory,
	// since muzoo test runs against HEAD in worktrees and won't see them.
	if repoRoot, err := gitRepoRoot(); err == nil {
		absMutDir, _ := filepath.Abs(mutDir)
		relMutDir, _ := filepath.Rel(repoRoot, absMutDir)
		exclude := fmt.Sprintf(":(exclude)%s", relMutDir)
		if diff, err := gitOutputDir(repoRoot, "diff", "HEAD", "--", ".", exclude); err == nil && diff != "" {
			fmt.Fprintf(os.Stderr, "muzoo: warning: working tree has uncommitted changes; tests run against HEAD\n")
		}
	}

	// Use git common dir parent for worktree placement.
	wtRoot, err := worktreeRoot()
	if err != nil {
		return fmt.Errorf("finding repository root: %w", err)
	}

	if err := ensureWorktreeParent(wtRoot); err != nil {
		return fmt.Errorf("creating worktree directory: %w", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-sigCh
		cancel()
	}()

	// Create or reuse worker worktrees, named by worker slot so the Go
	// build/test cache (keyed by absolute path) is shared across mutations.
	// Worktrees are kept across runs to preserve tool-managed directories
	// like .venv that are expensive to rebuild.
	workerPaths := make([]string, *jobs)
	for i := range workerPaths {
		workerPaths[i] = worktreeDir(wtRoot, strconv.Itoa(i))
		if err := reuseOrCreateWorktree(workerPaths[i]); err != nil {
			for j := range i {
				removeWorktree(workerPaths[j])
			}
			return fmt.Errorf("creating worktree: %w", err)
		}
	}

	// Disable Python bytecode caching. Python's .pyc validation uses
	// (mtime-seconds, file-size) pairs, not content hashes. Since git
	// checkout and git apply can produce files with the same mtime (same
	// second) and size as a previous mutation, Python may silently reuse
	// stale bytecode from a different mutation, causing false kills.
	testEnv := append(os.Environ(), "PYTHONDONTWRITEBYTECODE=1")

	// Run the test command on a clean worktree first to make sure it passes
	// without any mutations. If the tests are already broken, every mutation
	// would appear killed, giving a false positive.
	fmt.Fprintf(os.Stderr, "muzoo: running tests on un-mutated tree...\n")
	{
		wtPath := workerPaths[0]
		var sanityCmd *exec.Cmd
		if *timeout > 0 {
			tctx, tcancel := context.WithTimeout(ctx, *timeout)
			defer tcancel()
			sanityCmd = exec.CommandContext(tctx, "sh", "-c", strings.Join(testCmd, " "))
		} else {
			sanityCmd = exec.CommandContext(ctx, "sh", "-c", strings.Join(testCmd, " "))
		}
		sanityCmd.Dir = filepath.Join(wtPath, relDir)
		sanityCmd.Env = append(testEnv,
			"MUZOO_PATCH=",
			"MUZOO_DESCRIPTION=",
		)
		sanityCmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
		sanityCmd.Cancel = func() error {
			syscall.Kill(-sanityCmd.Process.Pid, syscall.SIGKILL)
			return nil
		}
		sanityCmd.WaitDelay = time.Second
		var outBuf bytes.Buffer
		sanityCmd.Stdout = &outBuf
		sanityCmd.Stderr = &outBuf
		oom, err := runCapped(sanityCmd, memLimit)
		if oom {
			return &exitError{code: 2, msg: fmt.Sprintf(
				"baseline tests exceeded the %s memory limit; raise or unset -memory", *memory)}
		}
		if err != nil {
			if ctx.Err() != nil {
				return &exitError{code: 2, msg: "interrupted"}
			}
			output := outBuf.String()
			if defaultGoTest {
				output = formatGoTestOutput(output)
			}
			fmt.Fprintf(os.Stderr, "\n%s\n", output)
			return &exitError{code: 2, msg: "tests fail without mutations; fix the tests first"}
		}
		fmt.Fprintf(os.Stderr, "muzoo: tests pass, running mutations...\n")
	}

	// Pre-read and validate all patches against a clean worktree (at HEAD),
	// not the user's potentially-dirty working tree.
	type patchInfo struct {
		name string
		desc string
		diff string
	}
	var infos []patchInfo
	for _, p := range patches {
		desc, diff, err := readPatch(mutDir, p)
		if err != nil {
			return fmt.Errorf("reading %s: %w", p, err)
		}
		if err := gitApplyCheck(workerPaths[0], diff); err != nil {
			return &exitError{code: 2, msg: fmt.Sprintf("patch %s does not apply cleanly; run 'muzoo rebase' first", p)}
		}
		infos = append(infos, patchInfo{name: p, desc: desc, diff: diff})
	}

	type result struct {
		patch       string
		desc        string
		survived    bool
		errored     bool
		timedOut    bool
		oomKilled   bool
		output      string
		killedTests string
	}

	results := make([]result, len(infos))
	// Pre-populate results so cancelled goroutines still have names.
	for i, info := range infos {
		results[i] = result{patch: info.name, desc: descriptionLabel(info.desc)}
	}

	// Worker pool: each slot is a worktree index.
	sem := make(chan int, *jobs)
	for i := range *jobs {
		sem <- i
	}
	var wg sync.WaitGroup

	testCmdStr := strings.Join(testCmd, " ")

	bar := progressbar.NewOptions(len(infos),
		progressbar.OptionSetWriter(os.Stderr),
		progressbar.OptionSetDescription("testing"),
		progressbar.OptionShowCount(),
		progressbar.OptionClearOnFinish(),
		progressbar.OptionSetElapsedTime(true),
		progressbar.OptionShowElapsedTimeOnFinish(),
	)

	var runningMu sync.Mutex
	running := make(map[string]time.Time)
	updateBarDesc := func() {
		var names []string
		for n := range running {
			names = append(names, n)
		}
		sort.Strings(names)
		for i, n := range names {
			if d := time.Since(running[n]); d > 10*time.Second {
				names[i] = fmt.Sprintf("%s (%ds)", n, int(d.Seconds()))
			}
		}
		bar.Describe("testing " + strings.Join(names, ", "))
	}

	// Refresh the bar every second so per-mutation elapsed times and the
	// overall timer keep moving while slow mutations run.
	tickDone := make(chan struct{})
	go func() {
		t := time.NewTicker(time.Second)
		defer t.Stop()
		for {
			select {
			case <-tickDone:
				return
			case <-t.C:
				runningMu.Lock()
				updateBarDesc()
				runningMu.Unlock()
			}
		}
	}()

	for i, info := range infos {
		wg.Add(1)
		go func(idx int, info patchInfo) {
			defer wg.Done()

			num := strings.TrimSuffix(info.name, ".patch")

			var worker int
			select {
			case <-ctx.Done():
				return
			case worker = <-sem:
			}
			defer func() { sem <- worker }()

			runningMu.Lock()
			running[num] = time.Now()
			updateBarDesc()
			runningMu.Unlock()
			defer func() {
				runningMu.Lock()
				delete(running, num)
				updateBarDesc()
				runningMu.Unlock()
				bar.Add(1)
			}()

			wtPath := workerPaths[worker]

			// Reset worktree to clean state for this mutation.
			if err := resetWorktree(wtPath); err != nil {
				results[idx].errored = true
				results[idx].output = "worktree reset failed: " + err.Error()
				return
			}

			// Apply patch.
			if err := gitApply(wtPath, info.diff); err != nil {
				results[idx].errored = true
				results[idx].output = "apply failed: " + err.Error()
				return
			}

			// Run test command. Create the timeout context here (not
			// earlier) so the timeout covers only test execution, not
			// worktree reset and patch application.
			cmdCtx := ctx
			if *timeout > 0 {
				var tcancel context.CancelFunc
				cmdCtx, tcancel = context.WithTimeout(ctx, *timeout)
				defer tcancel()
			}
			cmd := exec.CommandContext(cmdCtx, "sh", "-c", testCmdStr)
			cmd.Dir = filepath.Join(wtPath, relDir)
			cmd.Env = append(testEnv,
				"MUZOO_PATCH="+info.name,
				"MUZOO_DESCRIPTION="+firstLine(info.desc),
			)
			// Use a process group so we can kill child processes on
			// timeout or signal, preventing orphaned test processes.
			cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
			cmd.Cancel = func() error {
				syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
				return nil
			}
			cmd.WaitDelay = time.Second
			var outBuf bytes.Buffer
			cmd.Stdout = &outBuf
			cmd.Stderr = &outBuf

			oom, err := runCapped(cmd, memLimit)
			output := outBuf.String()
			if defaultGoTest {
				output = formatGoTestOutput(output)
			} else if pytestCmd {
				output = formatPytestOutput(output)
			}
			if err == nil {
				// exit 0 = tests passed = mutation survived (BAD)
				results[idx].survived = true
				results[idx].output = output
			} else if ctx.Err() != nil {
				// Parent context cancelled (SIGINT/SIGTERM).
				return
			} else if cmdCtx.Err() == context.DeadlineExceeded {
				// Timeout expired = mutation killed (GOOD).
				results[idx].timedOut = true
				results[idx].output = output
			} else if oom {
				// Exceeded the memory limit = mutation killed (GOOD).
				results[idx].oomKilled = true
				results[idx].output = output
			} else {
				var exitErr *exec.ExitError
				if errors.As(err, &exitErr) &&
					exitErr.ExitCode() != 126 && exitErr.ExitCode() != 127 {
					// Non-zero exit = tests failed = mutation killed (GOOD).
					results[idx].output = output
					if defaultGoTest {
						results[idx].killedTests = formatFailedTests(
							parseFailedTests(outBuf.String()), 3)
					} else if pytestCmd {
						results[idx].killedTests = formatFailedTests(
							parsePytestFailedTests(outBuf.String()), 1)
					}
				} else {
					// Infrastructure error: either not an ExitError (e.g.
					// working directory doesn't exist) or shell exit 126/127
					// (command not found or not executable).
					results[idx].errored = true
					results[idx].output = output + err.Error()
				}
			}
		}(i, info)
	}

	wg.Wait()
	close(tickDone)

	signal.Stop(sigCh)

	// If interrupted, don't print a misleading partial summary.
	if ctx.Err() != nil {
		return &exitError{code: 2, msg: "interrupted"}
	}

	// Print results.
	tty := isTerminal(os.Stdout)
	killed := 0
	survivedCount := 0
	errorCount := 0
	for _, r := range results {
		num := strings.TrimSuffix(r.patch, ".patch")
		switch {
		case r.errored:
			fmt.Printf("%s  %s     %s\n", num, colorize(tty, "ERROR", colorRed), r.desc)
			errorCount++
		case r.survived:
			fmt.Printf("%s  %s  %s\n", num, colorize(tty, "SURVIVED", colorRed), r.desc)
			survivedCount++
		case r.timedOut:
			fmt.Printf("%s  %s   %s\n", num, colorize(tty, "TIMEOUT", colorGreen), r.desc)
			killed++
		case r.oomKilled:
			fmt.Printf("%s  %s       %s\n", num, colorize(tty, "OOM", colorGreen), r.desc)
			killed++
		default:
			killedTests := colorize(tty, r.killedTests, colorDim)
			fmt.Printf("%s  %s    %s%s\n", num, colorize(tty, "KILLED", colorGreen), r.desc, killedTests)
			killed++
		}
	}

	// Print output for errored mutations, and killed if verbose.
	for _, r := range results {
		show := (r.errored || *verbose) && r.output != ""
		if show {
			fmt.Printf("\n--- Output for %s (%s) ---\n%s\n", strings.TrimSuffix(r.patch, ".patch"), r.desc, r.output)
		}
	}

	if survivedCount > 0 || errorCount > 0 {
		return &exitError{code: 1, msg: fmt.Sprintf("%d mutation(s) survived, %d errored, %d killed", survivedCount, errorCount, killed)}
	}
	fmt.Fprintf(os.Stderr, "muzoo: all %d mutation(s) killed\n", killed)
	return nil
}

// parseFailedTests extracts unique leaf failed test names from go test -json output.
func parseFailedTests(output string) []string {
	type testEvent struct {
		Action string `json:"Action"`
		Test   string `json:"Test"`
	}
	seen := make(map[string]bool)
	for _, line := range strings.Split(output, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "{") {
			continue
		}
		var ev testEvent
		if err := json.Unmarshal([]byte(line), &ev); err != nil {
			continue
		}
		if ev.Action == "fail" && ev.Test != "" {
			seen[ev.Test] = true
		}
	}
	// Filter to leaf tests only (exclude parents of subtests).
	var failed []string
	for t := range seen {
		isParent := false
		for t2 := range seen {
			if t2 != t && strings.HasPrefix(t2, t+"/") {
				isParent = true
				break
			}
		}
		if !isParent {
			failed = append(failed, t)
		}
	}
	sort.Strings(failed)
	return failed
}

// formatGoTestOutput extracts human-readable output from go test -json lines.
func formatGoTestOutput(output string) string {
	type testEvent struct {
		Action string `json:"Action"`
		Output string `json:"Output"`
	}
	var b strings.Builder
	for _, line := range strings.Split(output, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "{") {
			if line != "" {
				b.WriteString(line)
				b.WriteByte('\n')
			}
			continue
		}
		var ev testEvent
		if err := json.Unmarshal([]byte(line), &ev); err != nil {
			b.WriteString(line)
			b.WriteByte('\n')
			continue
		}
		if ev.Action == "output" {
			b.WriteString(ev.Output)
		}
	}
	return b.String()
}

// formatFailedTests returns a short summary of failed tests for display.
// maxShow controls how many test names to include before truncating.
func formatFailedTests(tests []string, maxShow int) string {
	if len(tests) == 0 {
		return ""
	}
	if len(tests) <= maxShow {
		return " [" + strings.Join(tests, ", ") + "]"
	}
	return fmt.Sprintf(" [%s, ... +%d more]", strings.Join(tests[:maxShow], ", "), len(tests)-maxShow)
}

// isPytestCmd returns true if the test command runs pytest.
func isPytestCmd(testCmd []string) bool {
	cmd := strings.Join(testCmd, " ")
	cmd = strings.TrimSpace(cmd)
	return cmd == "pytest" || cmd == "uv run pytest" ||
		strings.HasPrefix(cmd, "pytest ") || strings.HasPrefix(cmd, "uv run pytest ")
}

// addPytestFlags adds -v and --tb=short to a pytest command for parseable output.
func addPytestFlags(testCmd []string) []string {
	cmd := strings.Join(testCmd, " ")
	cmd = strings.TrimSpace(cmd)
	// Insert flags right after "pytest".
	if strings.HasPrefix(cmd, "uv run pytest") {
		cmd = strings.Replace(cmd, "uv run pytest", "uv run pytest -v --tb=short", 1)
	} else {
		cmd = strings.Replace(cmd, "pytest", "pytest -v --tb=short", 1)
	}
	return []string{cmd}
}

// parsePytestFailedTests extracts failed test names from pytest -v output.
// It looks for lines like "FAILED tests/test_foo.py::test_bar" in the
// short test summary section.
func parsePytestFailedTests(output string) []string {
	var failed []string
	for _, line := range strings.Split(output, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "FAILED ") {
			continue
		}
		// "FAILED tests/test_foo.py::TestClass::test_bar - reason"
		name := strings.TrimPrefix(line, "FAILED ")
		if dash := strings.Index(name, " - "); dash != -1 {
			name = name[:dash]
		}
		// Extract just the test function/method name (after last ::).
		if idx := strings.LastIndex(name, "::"); idx != -1 {
			name = name[idx+2:]
		}
		failed = append(failed, name)
	}
	sort.Strings(failed)
	return failed
}

// formatPytestOutput trims pytest output to the most useful parts.
// With -v --tb=short, the output is already fairly concise, so we
// just return it as-is.
func formatPytestOutput(output string) string {
	return output
}

// parseSize parses a human-readable byte size like "2GiB", "512m", or a bare
// byte count. An empty string (or "0") returns 0, meaning no limit. Units are
// powers of 1024; a trailing "b"/"ib" is optional and ignored.
func parseSize(s string) (int64, error) {
	s = strings.TrimSpace(s)
	if s == "" || s == "0" {
		return 0, nil
	}
	i := 0
	for i < len(s) && (s[i] == '.' || (s[i] >= '0' && s[i] <= '9')) {
		i++
	}
	num, err := strconv.ParseFloat(s[:i], 64)
	if err != nil {
		return 0, fmt.Errorf("invalid size %q", s)
	}
	unit := strings.ToLower(strings.TrimSpace(s[i:]))
	unit = strings.TrimSuffix(unit, "b")
	unit = strings.TrimSuffix(unit, "i")
	var mult int64
	switch unit {
	case "":
		mult = 1
	case "k":
		mult = 1 << 10
	case "m":
		mult = 1 << 20
	case "g":
		mult = 1 << 30
	case "t":
		mult = 1 << 40
	default:
		return 0, fmt.Errorf("invalid size unit in %q", s)
	}
	return int64(num * float64(mult)), nil
}

// runCapped runs cmd like cmd.Run, but if memLimit is positive it periodically
// samples the resident memory of cmd's process group and, when the total
// exceeds memLimit, kills the group and reports oom=true. It relies on the
// caller having set SysProcAttr.Setpgid, so cmd.Process.Pid is the group ID.
func runCapped(cmd *exec.Cmd, memLimit int64) (oom bool, err error) {
	if memLimit <= 0 {
		return false, cmd.Run()
	}
	if err := cmd.Start(); err != nil {
		return false, err
	}
	var killed atomic.Bool
	done := make(chan struct{})
	go func() {
		t := time.NewTicker(250 * time.Millisecond)
		defer t.Stop()
		for {
			select {
			case <-done:
				return
			case <-t.C:
				if processGroupRSS(cmd.Process.Pid) > memLimit {
					killed.Store(true)
					syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
					return
				}
			}
		}
	}()
	err = cmd.Wait()
	close(done)
	return killed.Load(), err
}

// processGroupRSS returns the total resident set size in bytes of all
// processes in the given process group, or 0 if it cannot be determined.
func processGroupRSS(pgid int) int64 {
	out, err := exec.Command("ps", "-A", "-o", "pgid=,rss=").Output()
	if err != nil {
		return 0
	}
	var total int64
	for _, line := range strings.Split(string(out), "\n") {
		fields := strings.Fields(line)
		if len(fields) != 2 {
			continue
		}
		pg, err1 := strconv.Atoi(fields[0])
		rssKB, err2 := strconv.Atoi(fields[1])
		if err1 != nil || err2 != nil || pg != pgid {
			continue
		}
		total += int64(rssKB) * 1024
	}
	return total
}
