    # api_key="AIzaSyB1wNDsbNI4kALatkoRYKaGwHfLLS2iGl0"
import speech_recognition as sr
import pyttsx3
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage

# Initialize text-to-speech
engine = pyttsx3.init()

# Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.3,
    api_key="AIzaSyB1wNDsbNI4kALatkoRYKaGwHfLLS2iGl0"
)

def speech_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎙️ Listening...")
        audio = r.listen(source)
    try:
        return r.recognize_google(audio)
    except sr.UnknownValueError:
        return ""
    except sr.RequestError:
        return ""

def response_to_speech(text):
    if not text.strip():
        print("⚠️ Empty input.")
        return
    try:
        result = llm.invoke(text)

        # ✅ Extract content if it's an AIMessage
        if isinstance(result, AIMessage):
            response = result.content
        else:
            response = str(result)

        print("🤖 Agent:", response)
        engine.say(response)  # Now this is a string
        engine.runAndWait()

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("✅ Voice Assistant started. Speak to interact.")
    while True:
        text = speech_to_text()
        print(f"You said: {text}")
        if text.lower().strip() in ["exit", "stop", "quit"]:
            print("Exiting...")
            break
        response_to_speech(text)
