(function () {
  try {
    var existing = document.querySelector('script[src*="buymeacoffee.com/1.0.0/button.prod.min.js"]');
    if (existing) return;

    var s = document.createElement("script");
    s.type = "text/javascript";
    s.src = "https://cdnjs.buymeacoffee.com/1.0.0/button.prod.min.js";
    s.setAttribute("data-name", "bmc-button");
    s.setAttribute("data-slug", "redlightsoff5");
    s.setAttribute("data-color", "#FFDD00");
    s.setAttribute("data-emoji", "🏎️");
    s.setAttribute("data-font", "Arial");
    s.setAttribute("data-text", "Apoya esta página");
    s.setAttribute("data-outline-color", "#000000");
    s.setAttribute("data-font-color", "#000000");
    s.setAttribute("data-coffee-color", "#ffffff");
    document.body.appendChild(s);
  } catch (e) {}
})();
