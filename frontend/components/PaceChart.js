"use client";

import React from "react";
import ReactECharts from "echarts-for-react";

export default function PaceChart({ x, series, hiddenDrivers, watermark }) {
  const legendSelected = {};
  (series || []).forEach(s => { legendSelected[s.name] = !hiddenDrivers?.has(s.name); });

  const option = {
    animation: true,
    backgroundColor: "transparent",
    grid: { left: 55, right: 22, top: 42, bottom: 64 },
    tooltip: { trigger: "axis" },
    legend: {
      type: "scroll",
      top: 6,
      selected: legendSelected
    },
    toolbox: { feature: { dataZoom: {}, restore: {}, saveAsImage: {} } },
    dataZoom: [
      { type: "inside", xAxisIndex: 0 },
      { type: "slider", xAxisIndex: 0, height: 22, bottom: 10 }
    ],
    xAxis: { type: "category", data: x || [], name: "Lap", nameLocation: "middle", nameGap: 32 },
    yAxis: { type: "value", name: "Lap time (s)" },
    graphic: watermark ? [{
      type: "text",
      right: 18,
      bottom: 34,
      style: { text: watermark, font: "700 12px ui-sans-serif", fill: "rgba(17,17,17,.35)" }
    }] : [],
    series: (series || []).map(s => ({
      name: s.name,
      type: "line",
      data: s.data,
      showSymbol: false,
      smooth: true,
      connectNulls: false,
      lineStyle: { width: 2, color: s.color || "#111" },
      emphasis: { focus: "series" }
    }))
  };

  return (
    <ReactECharts
      option={option}
      style={{ height: 520, width: "100%" }}
      notMerge={true}
      lazyUpdate={true}
    />
  );
}
