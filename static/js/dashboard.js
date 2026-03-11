(function () {
    if (!window.distributionData) {
        return;
    }

    const ctx = document.getElementById('distributionChart');
    if (!ctx) {
        return;
    }

    const labels = Object.keys(window.distributionData);
    const values = Object.values(window.distributionData);

    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels,
            datasets: [{
                data: values,
                backgroundColor: ['#1D6FFF', '#5EA0FF', '#9BC3FF', '#CFE1FF'],
                borderColor: '#ffffff',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            cutout: '60%',
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
})();
