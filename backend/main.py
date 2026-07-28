import json
import os
from typing import Any, Callable, Dict, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from groq import Groq

from core.confidence import is_low_confidence
from core.pose_service import extract_keypoints
from core.video_processor import process_video
from fallback.gemini_service import analyze_with_fallback
from modules import health_analysis, sprint_analysis, tennis_analysis


BASE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIRECTORY = os.path.abspath(
    os.path.join(BASE_DIRECTORY, "../frontend")
)

app = FastAPI(
    title="AI Sports and Healthcare Tracker",
    version="1.0.0",
)

# Serve the frontend only when the directory exists.
if os.path.isdir(FRONTEND_DIRECTORY):
    app.mount(
        "/static",
        StaticFiles(directory=FRONTEND_DIRECTORY),
        name="static",
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_groq_client() -> Optional[Groq]:
    """
    Create a Groq client only when an API key is configured.
    """
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


@app.get("/")
def serve_frontend():
    """
    Serve frontend/index.html when it exists.
    Otherwise return a basic API status response.
    """
    index_path = os.path.join(
        FRONTEND_DIRECTORY,
        "index.html",
    )

    if os.path.isfile(index_path):
        return FileResponse(index_path)

    return {
        "status": "ok",
        "message": "Movement Analysis API is running.",
    }


@app.get("/health")
def health_check():
    """
    Lightweight endpoint for Render health checks.
    """
    return {
        "status": "healthy",
    }


def translate_to_natural_language(
    data: Dict[str, Any],
    mode: str,
) -> str:
    """
    Convert structured analysis data into readable coaching advice.

    If Groq is unavailable, return the analysis module's own feedback.
    """
    fallback_feedback = str(
        data.get(
            "feedback",
            "Analysis completed.",
        )
    )

    client = get_groq_client()

    if client is None:
        return fallback_feedback

    prompt = f"""
You are a sports and health movement coach.

Convert the following structured analysis into short, clear,
and practical feedback.

Mode: {mode}

Analysis:
{json.dumps(data, indent=2)}

Requirements:
- Use plain language.
- Explain the most important finding.
- Give two or three actionable suggestions.
- Do not invent findings that are not present in the analysis.
- Do not diagnose a medical condition.
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.3,
        )

        generated_text = response.choices[0].message.content

        if generated_text:
            return generated_text

    except Exception as exc:
        print(
            f"Groq translation failed: {exc}",
            flush=True,
        )

    return fallback_feedback


def build_response(
    result: Dict[str, Any],
    mode: str,
    used_fallback: bool,
) -> Dict[str, Any]:
    """
    Always return the same response structure.

    This prevents the frontend from receiving undefined fields.
    """
    normalized_result = dict(result)

    normalized_result["mode"] = mode
    normalized_result["used_fallback"] = used_fallback
    normalized_result.setdefault("issues", [])
    normalized_result.setdefault(
        "feedback",
        "Analysis completed.",
    )

    return {
        "used_fallback": used_fallback,
        "json": normalized_result,
        "natural_language": translate_to_natural_language(
            normalized_result,
            mode,
        ),
    }


@app.post("/analyze")
async def analyze(
    mode: str,
    video: UploadFile = File(...),
):
    """
    Analyze an uploaded movement video.
    """
    normalized_mode = mode.strip().lower()

    analyzers: Dict[
        str,
        Callable[[list], Dict[str, Any]],
    ] = {
        "health": health_analysis.run,
        "sprint": sprint_analysis.run,
        "tennis": tennis_analysis.run,
    }

    if normalized_mode not in analyzers:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid analysis mode. "
                "Use health, sprint, or tennis."
            ),
        )

    filename = video.filename or ""

    print(
        f"Starting {normalized_mode} analysis for {filename}",
        flush=True,
    )

    try:
        contents = await video.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="The uploaded video is empty.",
            )

        print(
            f"Uploaded video size: {len(contents)} bytes",
            flush=True,
        )

        # Keep memory usage low on Render.
        frames = process_video(
            contents,
            target_fps=3,
            target_height=256,
            max_frames=40,
        )

        del contents

        if not frames:
            raise HTTPException(
                status_code=400,
                detail=(
                    "The video could not be decoded. "
                    "Please upload a short MP4 video encoded with H.264."
                ),
            )

        frame_count = len(frames)

        print(
            f"Frames extracted: {frame_count}",
            flush=True,
        )

        keypoints = extract_keypoints(frames)

        # The current fallback does not inspect frame pixels,
        # so release the images immediately.
        del frames

        if not keypoints:
            fallback_result = analyze_with_fallback(
                frame_count=frame_count,
                mode=normalized_mode,
                reason="No pose data was returned.",
            )

            return build_response(
                fallback_result,
                normalized_mode,
                used_fallback=True,
            )

        detected_frame_count = sum(
            1 for frame in keypoints if frame
        )

        print(
            (
                f"Pose detected in {detected_frame_count} "
                f"of {len(keypoints)} frames"
            ),
            flush=True,
        )

        if detected_frame_count == 0:
            fallback_result = analyze_with_fallback(
                frame_count=frame_count,
                mode=normalized_mode,
                reason="No full-body pose was detected.",
            )

            return build_response(
                fallback_result,
                normalized_mode,
                used_fallback=True,
            )

        if is_low_confidence(keypoints):
            fallback_result = analyze_with_fallback(
                frame_count=frame_count,
                mode=normalized_mode,
                reason=(
                    "Pose landmark visibility was too low "
                    "for reliable angle analysis."
                ),
            )

            return build_response(
                fallback_result,
                normalized_mode,
                used_fallback=True,
            )

        analysis_function = analyzers[normalized_mode]
        result = analysis_function(keypoints)

        print(
            f"{normalized_mode} analysis completed",
            flush=True,
        )

        return build_response(
            result,
            normalized_mode,
            used_fallback=False,
        )

    except HTTPException:
        raise

    except Exception as exc:
        print(
            f"Unexpected analysis error: {repr(exc)}",
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {exc}",
        ) from exc

    finally:
        await video.close()