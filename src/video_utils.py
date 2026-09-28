"""Helpers to make OpenCV output playable inside a web browser (H.264 + yuv420p)."""
import shutil
import subprocess
from pathlib import Path


def _ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def make_browser_playable(src, dst):
    """Re-encode `src` to H.264 MP4 at `dst`. Returns True if re-encoded.
    Falls back to simply moving the file if ffmpeg is unavailable."""
    src, dst = Path(src), Path(dst)
    exe = _ffmpeg_exe()
    if exe:
        cmd = [exe, "-y", "-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-preset", "veryfast", "-crf", "23", "-movflags", "+faststart",
               "-an", str(dst)]
        try:
            r = subprocess.run(cmd, capture_output=True)
            if r.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
                src.unlink(missing_ok=True)
                return True
        except Exception:
            pass
    shutil.move(str(src), str(dst))
    return False
