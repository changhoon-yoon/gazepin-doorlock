# Beginner-friendly launcher for the packaged (PyInstaller) builds.
# Keeps relative paths working by moving to the executable's folder when frozen.
import os
import sys


def main():
    if getattr(sys, "frozen", False):
        os.chdir(os.path.dirname(sys.executable))
    print("=" * 50)
    print("  GazePIN Doorlock demos")
    print("=" * 50)
    print("  1) HeadPIN - head-turn 4-way, PIN 0-9  (recommended)")
    print("  2) GazePIN - eye-gaze 2-way, symbols 1-8")
    choice = input("Select demo [1/2] (default 1): ").strip() or "1"
    pin = input("PIN (default 1234): ").strip() or "1234"
    sys.argv = [sys.argv[0], "--pin", pin]
    try:
        if choice == "2":
            import gazepin_demo
            gazepin_demo.main()
        else:
            import headpin_demo
            headpin_demo.main()
    except Exception:
        import traceback
        traceback.print_exc()
        input("\nAn error occurred (see above). Press Enter to close...")


if __name__ == "__main__":
    main()
