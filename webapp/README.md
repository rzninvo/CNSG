# Web Application

This web application is a frontend project built with Vite, React, and TypeScript.  
It is intended to be run locally for development and evaluation.

## Requirements

- Node.js (>=18 recommended)
- npm

To manage Node versions, you may use nvm:  
https://github.com/nvm-sh/nvm#installing-and-updating

## Local Setup and Development

Clone the repository and install dependencies:

```sh
# Clone the repository
git clone <REPOSITORY_URL>

# Navigate into the project directory
cd webapp

# Install dependencies
npm install
```

Start the development server:

```sh
npm run dev
```

By default, the application runs on:

- http://localhost:8080

The development server supports hot-reloading, so changes are reflected immediately in the browser.

## Accessing the Web App from a Mobile Device (ngrok)

To test the application from a smartphone (or any external device), you can expose the local development server using **ngrok**.

1. Install ngrok:  
   https://ngrok.com/download

2. Authenticate ngrok (required once):

```sh
ngrok config add-authtoken <YOUR_NGROK_TOKEN>
```

3. Expose the local server on port 8080:

```sh
ngrok http 8080
```

4. ngrok will output a public HTTPS URL (e.g., `https://xxxx.ngrok.io`).  
   Open that URL from your phone to access the running web application.

This setup was used to run the frontend locally and access it from a mobile device.

## Editing the Code

You can edit the code using:

- Any local IDE (e.g., VS Code, IntelliJ)
- Direct edits on GitHub (for small changes)
- GitHub Codespaces (optional)

Local development is recommended for debugging and iterative work.

## Technologies Used

- Vite
- TypeScript
- React
- Tailwind CSS
- shadcn/ui

## Notes

- The application is intended to be run locally.
- No proprietary platforms or external services are required to run it.
- Everything needed to run the web application is contained in this repository.
