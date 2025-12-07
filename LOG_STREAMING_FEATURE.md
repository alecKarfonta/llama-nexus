# Log Streaming Feature

## Overview
Added real-time log streaming from the deployed Docker container to the Deploy page. This helps with debugging and monitoring the model inference service.

## Implementation

### Backend Changes (`backend/main.py`)

#### New Endpoints:

1. **GET `/api/v1/logs/container?lines=N`**
   - Fetches the last N lines of logs from the Docker container
   - Returns JSON with log entries
   - Example:
     ```bash
     curl http://localhost:8700/api/v1/logs/container?lines=100
     ```

2. **GET `/api/v1/logs/container/stream`**
   - Streams logs in real-time using Server-Sent Events (SSE)
   - Automatically follows new log entries as they appear
   - Client can connect and receive continuous updates

### Frontend Changes

#### New Component: `LogViewer.tsx`

A full-featured log viewer component with:

- **Real-time Streaming**: Uses EventSource API to connect to SSE endpoint
- **Auto-scroll**: Automatically scrolls to bottom as new logs arrive (toggleable)
- **Play/Pause**: Control streaming with a button
- **Refresh**: Manually reload logs
- **Clear**: Clear the log display
- **Download**: Export logs to a text file
- **Line Limit**: Keeps only the last 500 lines (configurable)
- **Syntax Highlighting**: Terminal-style display with dark theme
- **Timestamps**: Shows timestamp for each log entry

#### Deploy Page Integration

- Log viewer appears at the bottom of the Deploy page
- Only visible when a service is running
- Contained in a Card with header
- Height set to 500px with scrolling

## Features

### Log Viewer Controls

| Control | Icon | Function |
|---------|------|----------|
| Play/Pause | ▶️/⏸️ | Start or stop log streaming |
| Refresh | 🔄 | Reload logs from container |
| Clear | 🗑️ | Clear the log display |
| Download | 💾 | Export logs to text file |
| Auto-scroll | Toggle | Enable/disable auto-scrolling |

### Visual Features

- **Terminal Theme**: Dark background (#1e1e1e) with light text
- **Hover Effects**: Slight highlight on log lines
- **Scrollbar**: Custom styled scrollbar
- **Live Indicator**: Green "LIVE" chip when streaming
- **Line Count**: Shows number of log lines displayed
- **Timestamps**: Each line prefixed with time

## Usage

### For Users

1. Navigate to the Deploy page at http://localhost:3002/deploy
2. If a service is running, the log viewer appears at the bottom
3. Click the ▶️ Play button to start streaming logs
4. Watch real-time logs appear as the model processes requests
5. Use the download button to save logs for later analysis

### For Developers

**Fetch logs programmatically:**
```javascript
const response = await fetch('/api/v1/logs/container?lines=100')
const data = await response.json()
console.log(data.logs)
```

**Stream logs:**
```javascript
const eventSource = new EventSource('/api/v1/logs/container/stream')
eventSource.onmessage = (event) => {
  const log = JSON.parse(event.data)
  console.log(log.message)
}
```

## Technical Details

### Server-Sent Events (SSE)

- Chosen over WebSocket for simplicity (one-way communication)
- Automatic reconnection handled by browser
- HTTP-based, works through proxies
- No special server requirements

### Implementation Details

- Backend uses `asyncio.create_subprocess_exec` to run `docker logs -f`
- Logs are streamed line-by-line to prevent buffering
- Small 10ms delay between lines to prevent overwhelming the client
- Frontend uses `EventSource` API for SSE connection
- Logs are kept in a circular buffer (last 500 lines)

### Error Handling

- Connection errors display in the log viewer
- Failed API calls show error messages
- Graceful degradation if container is not running

## Files Created/Modified

### Created:
- `frontend/src/components/LogViewer.tsx` (315 lines)
- `LOG_STREAMING_FEATURE.md` (this file)

### Modified:
- `backend/main.py` - Added two new endpoints (lines ~2050-2110)
- `frontend/src/pages/DeployPage.tsx` - Integrated LogViewer component

## Benefits

1. **Real-time Debugging**: See errors as they happen
2. **No SSH Required**: View logs directly in the web UI
3. **Easy Export**: Download logs for sharing or analysis
4. **Historical View**: See past logs before starting stream
5. **Resource Efficient**: Only streams when actively viewing

## Future Enhancements

Possible improvements:
- Add log filtering (search, regex)
- Color coding for error/warning/info levels
- Log level filtering
- Multiple container support
- Log rotation and archival
- Tail multiple files
- Syntax highlighting for JSON logs
- Collapsible stack traces

## Testing

The feature has been tested with:
- ✅ Static log fetching (`/api/v1/logs/container`)
- ✅ Real-time log streaming (`/api/v1/logs/container/stream`)
- ✅ Play/Pause controls
- ✅ Auto-scroll toggle
- ✅ Log download
- ✅ Clear functionality
- ✅ Refresh button

## Example Output

When working, the log viewer shows lines like:
```
20:33:45 srv  log_server_r: request: GET /health 127.0.0.1 200
20:33:45 srv  log_server_r: response: {"status":"ok"}
20:33:46 main: processing request...
20:33:47 srv  inference complete in 1.2s
```

## Troubleshooting

**No logs appearing:**
- Check if the container is running: `docker ps`
- Verify backend is accessible: `curl http://localhost:8700/health`
- Hard refresh browser (Ctrl+Shift+R)

**Logs stop streaming:**
- Click the Play button again
- Check browser console for EventSource errors
- Verify backend container is running

**High memory usage:**
- Logs are limited to 500 lines
- Clear logs periodically
- Pause streaming when not needed

