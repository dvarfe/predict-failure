function createTrace(deviceData, index, selectedFeature, metadata) {
    if (!deviceData.timestamps || deviceData.timestamps.length === 0) return null;

    return {
        x: deviceData.timestamps.map(ts => new Date(ts * 1000)),
        y: deviceData.values,
        type: 'scatter',
        mode: 'lines+markers',
        name: deviceData.device_name || deviceData.device_id,
        connectgaps: true,
        line: {
            width: 2
        },
        marker: {
            size: 4
        },
        hovertemplate: '<b>%{fullData.name}</b><br>' +
            'Время: %{x}<br>' +
            selectedFeature + ': %{y}' + (metadata?.unit ? ' ' + metadata.unit : '') + '<br>' +
            '<extra></extra>'
    };
}

function getPlotlyLayout(selectedFeature, metadata) {
    const yTitle = metadata?.description || selectedFeature;
    const yTitleWithUnit = yTitle + (metadata?.unit ? ` (${metadata.unit})` : '');
    const chartTitle = metadata?.description || selectedFeature;

    return {
        title: {
            text: chartTitle,
            font: { color: '#eee', size: 16 }
        },
        xaxis: {
            title: { text: 'Время', font: { color: '#eee' } },
            tickfont: { color: '#eee' },
            gridcolor: '#444',
            showgrid: true
        },
        yaxis: {
            title: { text: yTitleWithUnit, font: { color: '#eee' } },
            tickfont: { color: '#eee' },
            gridcolor: '#444',
            showgrid: true,
            tickformat: metadata?.unit === 'bytes' ? '.2s' : undefined 
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        legend: {
            font: { color: '#eee' },
            bgcolor: 'rgba(0,0,0,0.1)'
        },
        margin: { l: 80, r: 30, t: 50, b: 60 },
        hovermode: 'closest'
    };
}

const PLOTLY_CONFIG = {
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
    responsive: true
};

function createFeatureChart(chartData, selectedFeature) {
    console.log(chartData);
    if (chartData && chartData.length > 0) {
        const metadata = typeof (featureMetadata) !== 'undefined' ? featureMetadata : {};

        const traces = chartData
            .map((deviceData, index) => createTrace(deviceData, index, selectedFeature, metadata))
            .filter(trace => trace !== null);

        const layout = getPlotlyLayout(selectedFeature, metadata);
        const config = PLOTLY_CONFIG;

        Plotly.newPlot('featureChart', traces, layout, config);
    }
}

// Автоматически строим график при загрузке страницы
if (typeof chartData !== 'undefined' && typeof selectedFeature !== 'undefined') {
    createFeatureChart(chartData, selectedFeature);
}