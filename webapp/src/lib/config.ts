// Configuration for external services
export const config = {
  // Your Google Apps Script Web App URL
  googleSheetsUrl:
    "https://script.google.com/macros/s/AKfycbw4AzZeqoV9TgS8m6ut9qMLi_WR2TG_7TMNb53MzOCqe4hLx6JoPlih4r2qTHOLAAMn/exec",

  // Your ngrok server URL
  //serverUrl: "https://maddie-interlabial-jordy.ngrok-free.dev/process",
  serverUrl: "https://monasterial-daine-swirlier.ngrok-free.dev/process",
} as const;

// Debug: log della configurazione al caricamento
console.log("Config loaded:", {
  googleSheetsUrl: config.googleSheetsUrl,
  serverUrl: config.serverUrl,
});
