from dataclasses import dataclass
from typing import Optional, Callable
import asyncio

@dataclass
class ObstacleConfig:
    left_thresh: float = 0.3
    right_thresh: float = 0.3
    rear_thresh: float = 0.3
    consecutive_hits: int = 1


async def stop_on_obstacle(
    robot,
    phy,
    stop_evt: asyncio.Event,
    cfg: ObstacleConfig,
    *,
    on_trigger: Optional[Callable[[object], None]] = None,
    recovery_policy=None
) -> None:

    # Recovery policy
    if not recovery_policy:
        recovery_policy = recovery.StopRecoveryPolicy()

    st = phy.createStream()
    hits = 0
    try:
        while not stop_evt.is_set():
            s = await st.next()

            left = getattr(s, "left", 0.0)
            right = getattr(s, "right", 0.0)
            rear = getattr(s, "rear", 0.0)

            # print(f"left={left}, right={right}, rear={rear}")

            too_close = (
                (left is not None and left >= cfg.left_thresh) or
                (right is not None and right >= cfg.right_thresh) or
                (rear is not None and rear >= cfg.rear_thresh)
            )

            hits = hits + 1 if too_close else 0

            # Too many hits
            if hits >= cfg.consecutive_hits:
                
                # Apply Trigger
                if on_trigger:
                    try:
                        on_trigger(s)
                    except Exception:
                        pass

                # Try to recover
                try:
                    if recovery_policy:
                        await recovery_policy.handle_obstacle(robot, s)

                # If recovery fails, stop.    
                except Exception as e:
                    print(f"[safety] recovery policy failed: {e!r}")
                    try:
                        await robot.stop()
                    except Exception:
                        pass

                # Hold stop until cleared
                while not stop_evt.is_set():
                    s2      = await st.next()
                    left2   = getattr(s2, "left", 0.0)
                    right2  = getattr(s2, "right", 0.0)
                    rear2   = getattr(s2, "rear", 0.0)

                    still_close = (
                        (left2 is not None and left2 >= cfg.left_thresh) or
                        (right2 is not None and right2 >= cfg.right_thresh) or
                        (rear2 is not None and rear2 >= cfg.rear_thresh)
                    )

                    try:
                        await robot.stop()
                    except Exception:
                        pass

                    if not still_close:
                        hits = 0
                        break

    except asyncio.CancelledError:
        try:
            await robot.stop()
        except Exception:
            pass
        raise
    finally:
        await st.close()