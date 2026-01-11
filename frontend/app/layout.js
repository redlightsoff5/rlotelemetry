import "./globals.css";

export const metadata = {
  title: "RLO Telemetry",
  description: "F1 telemetry insights with ECharts",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <div className="appShell">
          <header className="topbar">
            <div className="brand">
              <div className="logoDot" />
              <div>
                <div className="brandTitle">RLO Telemetry</div>
                <div className="brandSub">@redlightsoff5</div>
              </div>
            </div>
            <div className="topActions">
              <a className="ghostBtn" href="https://www.buymeacoffee.com/" target="_blank" rel="noreferrer">Support</a>
            </div>
          </header>
          <main className="main">{children}</main>
          <footer className="footer">Data via FastF1. Visuals by RLO.</footer>
        </div>
      </body>
    </html>
  );
}
