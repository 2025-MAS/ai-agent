import os
from dotenv import load_dotenv
from services.agent_runner import ask_agent, ask_vision

load_dotenv()

# Default Vision settings
FRAME_INTERVAL_SECONDS = 2
VIDEO_PATH = "input/sample.mp4"  # Default video file path


def is_vision_question(text: str) -> bool:
    """Check whether the user message is a vision-related question."""
    vision_keywords = [
        "see", "what do you see", "what can you see", "show me",
        "describe the video", "describe the scene", "what is on the screen",
        "video", "frame", "scene"
    ]
    return any(keyword in text.lower() for keyword in vision_keywords)


def main():
    print("=== Starting AI Assistant Chat ===")
    print("Type your question below. (Type 'exit chat' to quit)\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "exit chat":
            print("AI: Chat ended. Have a great day!")
            break

        try:
            # Check if the question is related to Vision tasks
            if is_vision_question(user_input):
                print("AI: Analyzing video... please wait.\n")
                answer = ask_vision(
                    question=user_input,
                    video_path=VIDEO_PATH,
                    frame_interval=FRAME_INTERVAL_SECONDS
                )
                print("AI:", answer)

            else:
                # Otherwise, send to Weather Agent
                answer = ask_agent(user_input)
                print("AI:", answer)

        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    main()
