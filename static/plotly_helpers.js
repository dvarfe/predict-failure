(function (window) {
    const DEFAULT_CONFIG = {
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
        responsive: true
    };

    function getBaseDarkLayout() {
        return {
            title: { text: '', font: { color: '#eee', size: 16 } },
            xaxis: { title: { text: 'Время', font: { color: '#eee' } }, tickfont: { color: '#eee' }, gridcolor: '#444', showgrid: true },
            yaxis: { title: { text: '', font: { color: '#eee' } }, tickfont: { color: '#eee' }, gridcolor: '#444', showgrid: true },
            plot_bgcolor: 'rgba(0,0,0,0)',
            paper_bgcolor: 'rgba(0,0,0,0)',
            legend: { font: { color: '#eee' }, bgcolor: 'rgba(0,0,0,0.1)' },
            margin: { l: 80, r: 30, t: 50, b: 60 },
            hovermode: 'closest'
        };
    }

    function tracesFromCurves(curves) {
        const traces = [];
        if (!curves) return traces;
        for (const id in curves) {
            const c = curves[id];
            traces.push({ x: c.time, y: c.survival, mode: 'lines', name: String(id), line: { width: 2 } });
        }
        return traces;
    }

    function getDefaultSurvivalLayout(title) {
        const layout = getBaseDarkLayout();
        layout.title.text = title || 'Кривые выживания';
        layout.yaxis.title.text = 'Вероятность выживания';
        layout.yaxis.range = [0, 1];
        return layout;
    }

    function plotTraces(elementId, traces, layout, config) {
        const cfg = config || DEFAULT_CONFIG;
        Plotly.newPlot(elementId, traces, layout || getBaseDarkLayout(), cfg);
    }

    function plotSurvival(elementId, curves, title) {
        const traces = tracesFromCurves(curves);
        const layout = getDefaultSurvivalLayout(title);
        plotTraces(elementId, traces, layout, DEFAULT_CONFIG);
    }

    window.plotlyHelpers = {
        tracesFromCurves,
        getDefaultSurvivalLayout,
        plotTraces,
        plotSurvival,
        DEFAULT_CONFIG
    };
})(window);
