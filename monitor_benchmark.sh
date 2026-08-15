#!/bin/bash
# Monitor the benchmark progress in tmux

echo "📊 DCF Benchmark Monitor"
echo "Attach to the tmux session: tmux attach -t benchmark"
echo ""

# Show current status every 10 seconds
while true; do
    clear
    echo "═══════════════════════════════════════════════════════════════"
    echo "  DCF Benchmark Progress ($(date '+%Y-%m-%d %H:%M:%S'))"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""

    # Get the latest 15 lines from the tmux session
    tmux capture-pane -t benchmark -p | tail -15

    echo ""
    echo "─────────────────────────────────────────────────────────────"
    echo "To stop monitoring: Ctrl+C"
    echo "To interact with the session: tmux send-keys -t benchmark '<key>'"
    echo "═══════════════════════════════════════════════════════════════"

    sleep 10
done
