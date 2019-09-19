from app import app
import os

# app = create_app(os.getenv('FLASK_CONFIG') or 'default')


def main():
    app.run(host="0.0.0.0", port=3001)


if __name__ == "__main__":
    main()
