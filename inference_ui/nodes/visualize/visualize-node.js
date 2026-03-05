// ============================================
// Visualize Node Type - displays images/charts
// ============================================

NodeRegistry.register('visualize', {
    label: 'Visualize',
    category: 'General',
    result: true,

    ports: [
        { name: 'in', dir: 'in', type: 'any' },
        { name: 'out', dir: 'out', type: 'any' }
    ],

    defaultConfig: {
        title: 'Visualize',
        tool: '',
        menuTag: { en: 'Chart', ko: '시각화', ja: 'チャート', zh: '图表', fr: 'Graphique', de: 'Diagramm', es: 'Gráfico', it: 'Grafico', pt: 'Gráfico', nl: 'Grafiek', ru: 'Диаграмма', ar: 'رسم بياني', hi: 'चार्ट', tr: 'Grafik', pl: 'Wykres', cs: 'Graf', sv: 'Diagram', da: 'Diagram', no: 'Diagram', fi: 'Kaavio', el: 'Γράφημα', hu: 'Diagram', ro: 'Grafic', uk: 'Діаграма', vi: 'Biểu đồ', th: 'แผนภูมิ', id: 'Bagan' },
        description: { en: 'Visualize data as charts, plots, or graphical outputs', ko: '데이터를 차트, 플롯 등으로 시각화', ja: 'データをチャート、プロット、またはグラフィカル出力として視覚化する', zh: '将数据可视化为图表、绘图或图形输出', fr: 'Visualiser les données sous forme de graphiques, tracés ou sorties graphiques', de: 'Daten als Diagramme, Plots oder grafische Ausgaben visualisieren', es: 'Visualizar datos como gráficos, diagramas o salidas gráficas', it: 'Visualizza i dati come grafici, diagrammi o output grafici', pt: 'Visualizar dados como gráficos, plotagens ou saídas gráficas', nl: 'Gegevens visualiseren als grafieken, plots of grafische uitvoer', ru: 'Визуализация данных в виде диаграмм, графиков или графических выходов', ar: 'تصور البيانات كرسوم بيانية أو مخططات أو مخرجات رسومية', hi: 'डेटा को चार्ट, प्लॉट या ग्राफ़िकल आउटपुट के रूप में विज़ुअलाइज़ करें', tr: 'Verileri grafikler, çizimler veya grafiksel çıktılar olarak görselleştir', pl: 'Wizualizuj dane jako wykresy, grafy lub wyjścia graficzne', cs: 'Vizualizovat data jako grafy, výkresy nebo grafické výstupy', sv: 'Visualisera data som diagram, plottar eller grafiska utdata', da: 'Visualiser data som diagrammer, plots eller grafiske output', no: 'Visualiser data som diagrammer, plott eller grafiske utdata', fi: 'Visualisoi tiedot kaavioina, kuvaajina tai graafisina tulosteina', el: 'Οπτικοποίηση δεδομένων ως γραφήματα, διαγράμματα ή γραφικές εξόδους', hu: 'Adatok megjelenítése diagramokként, ábrákként vagy grafikus kimenetként', ro: 'Vizualizează datele ca grafice, diagrame sau ieșiri grafice', uk: 'Візуалізація даних у вигляді діаграм, графіків або графічних виходів', vi: 'Trực quan hóa dữ liệu dưới dạng biểu đồ, đồ thị hoặc đầu ra đồ họa', th: 'แสดงข้อมูลเป็นแผนภูมิ พล็อต หรือผลลัพธ์กราฟิก', id: 'Visualisasikan data sebagai bagan, plot, atau keluaran grafis' },
        status: 'pending',
        stepNum: '',
        resultText: ''
    },

    render(node, helpers) {
        const el = document.createElement('div');
        const hasResult = node.resultText && node.resultText.length > 0;
        const bodyStyle = hasResult ? '' : ' style="display:none"';

        el.innerHTML = `
            <div class="ng-ports-in-row" data-node-id="${node.id}">
                <div class="ng-port ng-port-in" data-port-name="in" data-port-dir="in" data-port-type="any" data-node-id="${node.id}"></div>
            </div>
            <div class="ng-node-header">
                <span class="ng-node-num">${node.stepNum}</span>
                <span class="ng-node-title">${helpers.escapeHtml(node.title)}</span>
            </div>
            <div class="ng-node-body ng-visualize-body"${bodyStyle}>
                <div class="ng-visualize-output"></div>
            </div>
            <div class="ng-node-progress"></div>
            <div class="ng-ports-out-row" data-node-id="${node.id}">
                <div class="ng-port ng-port-out" data-port-name="out" data-port-dir="out" data-port-type="any" data-node-id="${node.id}"></div>
            </div>
        `;

        if (hasResult) {
            const output = el.querySelector('.ng-visualize-output');
            this._renderResult(output, node.resultText);
        }

        return el;
    },

    getDragHandle(el) {
        return el.querySelector('.ng-node-header');
    },

    _renderResult(container, resultData) {
        if (!container || !resultData) return;
        if (resultData.startsWith('data:image') || resultData.startsWith('http')) {
            container.innerHTML = `<img src="${resultData}" alt="Visualization">`;
        } else {
            container.textContent = resultData;
        }
    },

    updateResult(el, resultData) {
        const body = el.querySelector('.ng-visualize-body');
        const output = el.querySelector('.ng-visualize-output');
        if (!body || !output) return;

        if (resultData) {
            body.style.display = '';
            this._renderResult(output, resultData);
        } else {
            body.style.display = 'none';
            output.innerHTML = '';
        }
    }
});
