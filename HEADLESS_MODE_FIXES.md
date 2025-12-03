# Headless Mode Compatibility Fixes

This document summarizes all pygame subsystem fixes applied to enable headless training mode (render_mode="rgb_array").

## Overview

When running in headless mode, pygame subsystems (display, mixer, font, time) are not initialized. The codebase has been updated to gracefully handle missing subsystems while maintaining full functionality in human rendering mode.

## Fixed Pygame Subsystems

### 1. pygame.mixer (Sound System)

**Location**: `model/entity/entity.py`
**Issue**: Sound loading fails when mixer not initialized
**Solution**:

- Created `_DummySound` class with `play()` and `get_length()` methods
- Wrapped `SoundEffects.__init__` in try-except block
- Falls back to dummy sounds in headless mode

### 2. pygame.time (Timer System)

**Locations**:

- `model/entity/ghost/ghost.py` (line ~169)
- `draw/game_engine.py` (line ~107)

**Issue**: `pygame.time.wait()` and `pygame.time.set_timer()` fail without initialization
**Solution**:

- Wrapped timer calls in try-except blocks
- Skip timing operations in headless mode (they're only needed for rendering delays)

### 3. pygame.event (Event System)

**Location**: `rl/ghost_env.py` (line ~204), `rl/env.py`
**Issue**: `pygame.event.pump()` requires event system initialization
**Solution**:

- Made event.pump() conditional on `render_mode == "human"`
- Only pump events when actually rendering to screen

### 4. pygame.font (Font System)

**Location**: `draw/game_engine.py` (lines 40, 49)
**Issue**: `pygame.font.SysFont()` fails without font initialization
**Solution**:

- Created `_DummyFont` and `_DummySurface` classes
- Wrapped font initialization in try-except blocks
- Returns dummy font objects that create dummy surfaces for text rendering
- Text rendering is skipped in headless mode anyway (only happens in render_mode="human")

### 5. pygame.display (Display System)

**Locations**: `rl/ghost_env.py`, `rl/env.py`
**Status**: Already properly handled
**Implementation**:

- `pygame.init()` called but display not initialized in headless mode
- Create `pygame.Surface(RESOLUTION)` instead of `pygame.display.set_mode()` for headless
- `pygame.display.flip()` calls gated by `render_mode == "human"` checks

## Safe Pygame Operations in Headless Mode

The following pygame operations work correctly WITHOUT display initialization:

- ✅ `pygame.Surface()` - Create off-screen surfaces
- ✅ `pygame.draw.*` - All drawing operations work on surfaces
- ✅ `pygame.image.load()` - Image loading works fine
- ✅ `pygame.transform.*` - Transform operations (scale, rotate, flip)
- ✅ `pygame.surfarray.*` - Surface array operations for observation extraction
- ✅ `screen.blit()` - Blitting works on both display and regular surfaces

## Testing Headless Mode

To verify headless compatibility:

```bash
python train_ghost_agent.py --output runs/experiment01
```

The environment will:

1. Initialize pygame core (`pygame.init()`)
2. Create off-screen surface (`pygame.Surface(RESOLUTION)`)
3. Skip all display, sound, font, and timer operations
4. Run 50 parallel environments simultaneously
5. Extract observations via surfarray when needed

## Performance Benefits

Headless mode provides:

- **Faster Training**: No rendering overhead (~60 FPS limit removed)
- **Parallel Execution**: Run 50+ environments without display windows
- **Server Compatibility**: Train on remote servers without X11/display
- **Resource Efficiency**: No GPU/window manager overhead
