"use client";

import React, { useEffect, useMemo, useState } from "react";
import PaceChart from "../components/PaceChart";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE;

function uniq(arr){ return Array.from(new Set(arr)); }

export default function Page() {
  const [year, setYear] = useState(2025);
  const [event, setEvent] = useState("");
  const [session, setSession] = useState("Race");
  const [events, setEvents] = useState([]);
  const [meta, setMeta] = useState(null);
  const [pace, setPace] = useState(null);

  // no drivers selected by default: hide all in legend initially
  const [hiddenDrivers, setHiddenDrivers] = useState(new Set());
  const [pill, setPill] = useState("");

  // defaults
  useEffect(() => {
    (async () => {
      const r = await fetch(`${API_BASE}/default`);
      const d = await r.json();
      setYear(d.year);
      setEvent(d.event);
      setSession(d.session || "Race");
    })();
  }, []);

  // events list
  useEffect(() => {
    if (!year) return;
    (async () => {
      const r = await fetch(`${API_BASE}/events?year=${year}`);
      const d = await r.json();
      setEvents(d.events || []);
      if (!event && d.events?.length) setEvent(d.events[d.events.length - 1].EventName);
    })();
  }, [year]);

  // session meta
  useEffect(() => {
    if (!year || !event || !session) return;
    (async () => {
      const r = await fetch(`${API_BASE}/session/meta?year=${year}&event=${encodeURIComponent(event)}&session=${encodeURIComponent(session)}`);
      const d = await r.json();
      setMeta(d);
      // hide all drivers initially
      setHiddenDrivers(new Set(d.drivers || []));
      setPill("");
    })();
  }, [year, event, session]);

  // pace chart data
  useEffect(() => {
    if (!year || !event || !session) return;
    (async () => {
      const r = await fetch(`${API_BASE}/charts/pace?year=${year}&event=${encodeURIComponent(event)}&session=${encodeURIComponent(session)}&max_laps=80`);
      const d = await r.json();
      setPace(d);
    })();
  }, [year, event, session]);

  const topDrivers = useMemo(() => {
    if (!pace?.series) return [];
    // already sorted by best lap in API
    return pace.series.map(s => s.name);
  }, [pace]);

  function applySelection(n) {
    if (!meta?.drivers) return;
    const all = meta.drivers;
    let show = [];
    if (n === "all") show = all;
    if (n === "top5") show = topDrivers.slice(0, 5);
    if (n === "top10") show = topDrivers.slice(0, 10);

    // hiddenDrivers set = all - show
    const hidden = new Set(all.filter(d => !show.includes(d)));
    setHiddenDrivers(hidden);
    setPill(n);
  }

  return (
    <div className="grid">
      <section className="hero">
        <h1>Telemetry that looks like a product</h1>
        <p>Select a session, then click driver names/colors in the legend to compare. Use Top 5 / Top 10 / All to start fast.</p>
      </section>

      <section className="card">
        <div className="cardHeader">
          <div>
            <div className="cardTitle">Session</div>
            <div className="smallNote">Only completed events. Watermark: {meta?.watermark || "@redlightsoff5"}</div>
          </div>

          <div className="controls">
            <select value={year} onChange={(e) => setYear(parseInt(e.target.value, 10))}>
              <option value={2025}>2025</option>
              <option value={2026}>2026</option>
            </select>

            <select value={event} onChange={(e) => setEvent(e.target.value)}>
              {events.map((ev) => (
                <option key={ev.RoundNumber || ev.EventName} value={ev.EventName}>{ev.EventName}</option>
              ))}
            </select>

            <select value={session} onChange={(e) => setSession(e.target.value)}>
              {["FP1","FP2","FP3","Qualifying","Sprint","Sprint Qualifying","Race"].map(s => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>

            <div className="pills">
              <button className={"pill " + (pill==="top5" ? "pillActive" : "")} onClick={() => applySelection("top5")}>Top 5</button>
              <button className={"pill " + (pill==="top10" ? "pillActive" : "")} onClick={() => applySelection("top10")}>Top 10</button>
              <button className={"pill " + (pill==="all" ? "pillActive" : "")} onClick={() => applySelection("all")}>All</button>
            </div>
          </div>
        </div>

        <PaceChart
          x={pace?.x || []}
          series={pace?.series || []}
          hiddenDrivers={hiddenDrivers}
          watermark={meta?.watermark || "@redlightsoff5"}
        />
      </section>
    </div>
  );
}
