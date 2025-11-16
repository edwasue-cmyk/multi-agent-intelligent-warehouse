# Training History Duration Assessment

## Executive Summary

The Training History feature on the Forecasting page was evaluated to ensure training durations are properly captured and displayed. Issues were identified and fixed.

## Issues Identified

### 1. Duration Calculation Issue
**Problem**: Training sessions that completed in less than 60 seconds showed "0 min" duration.

**Root Cause**: 
- The duration calculation used `int((end_dt - start_dt).total_seconds() / 60)` which truncates to minutes
- A training that took 20 seconds would show as 0 minutes

**Example**:
- Training ID: `training_20251115_171101`
- Start: `2025-11-15T17:11:01.407798`
- End: `2025-11-15T17:11:21.014810`
- Actual Duration: ~20 seconds
- Displayed: "0 min" ❌

### 2. Duration Display Limitation
**Problem**: Frontend only displayed minutes, making it impossible to see durations under 1 minute.

## Fixes Implemented

### 1. Backend Duration Calculation (`src/api/routers/training.py`)

**Changes**:
- Calculate `duration_seconds` for accurate tracking
- Round to nearest minute for `duration_minutes` (minimum 1 minute for completed trainings)
- Store both `duration_minutes` and `duration_seconds` in training history

```python
# Calculate duration
start_dt = datetime.fromisoformat(start_time)
end_dt = datetime.fromisoformat(end_time)
duration_seconds = (end_dt - start_dt).total_seconds()
# Round to nearest minute (round up if >= 30 seconds, round down if < 30 seconds)
# But always show at least 1 minute for completed trainings that took any time
if duration_seconds > 0:
    duration_minutes = max(1, int(round(duration_seconds / 60)))
else:
    duration_minutes = 0

# Store both in training session
training_session = {
    ...
    "duration_minutes": duration_minutes,
    "duration_seconds": int(duration_seconds),  # Also store seconds for more accurate display
    ...
}
```

### 2. Frontend Duration Display (`src/ui/web/src/pages/Forecasting.tsx`)

**Changes**:
- Enhanced duration display to show seconds when available
- Format: 
  - `< 60 seconds`: "X sec"
  - `>= 60 seconds`: "Xm Ys" (e.g., "2m 10s")
  - Falls back to minutes if `duration_seconds` not available

```typescript
<TableCell>
  {(() => {
    // Use duration_seconds if available for more accurate display
    if (session.duration_seconds !== undefined) {
      const seconds = session.duration_seconds;
      if (seconds < 60) {
        return `${seconds} sec`;
      } else {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return secs > 0 ? `${mins}m ${secs}s` : `${mins} min`;
      }
    }
    // Fallback to duration_minutes
    return session.duration_minutes > 0 
      ? `${session.duration_minutes} min` 
      : '< 1 min';
  })()}
</TableCell>
```

### 3. TypeScript Interface Update (`src/ui/web/src/services/trainingAPI.ts`)

**Changes**:
- Added optional `duration_seconds` field to `TrainingHistory` interface

```typescript
export interface TrainingHistory {
  training_sessions: Array<{
    ...
    duration_minutes: number;
    duration_seconds?: number;  // Optional: more accurate duration in seconds
    ...
  }>;
}
```

## Test Results

### Before Fix
```
Training ID: training_20251115_171101
Duration: 0 min ❌ (actually ~20 seconds)
```

### After Fix
```
Training ID: training_20251115_171101
Duration: 20 sec ✅ (accurate)
```

### Duration Display Examples
- `19 seconds` → "19 sec"
- `65 seconds` → "1m 5s"
- `120 seconds` → "2 min" (no seconds when exactly on minute)
- `130 seconds` → "2m 10s"

## Verification

### API Response
```json
{
  "training_sessions": [
    {
      "id": "training_20251115_171101",
      "start_time": "2025-11-15T17:11:01.407798",
      "end_time": "2025-11-15T17:11:21.014810",
      "duration_minutes": 1,
      "duration_seconds": 20,
      "status": "completed"
    }
  ]
}
```

### Frontend Display
- Training sessions now show accurate durations
- Short trainings (< 1 minute) display in seconds
- Longer trainings display in minutes and seconds format

## Recommendations

1. ✅ **Duration Calculation**: Fixed - now accurately captures seconds
2. ✅ **Duration Display**: Fixed - shows seconds for short trainings
3. ✅ **Backward Compatibility**: Maintained - falls back to minutes if seconds not available
4. ⚠️ **Database Storage**: Consider storing training history in database instead of in-memory for persistence

## Status

✅ **RESOLVED** - Training duration is now properly captured and displayed with second-level accuracy for short trainings.

