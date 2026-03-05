// ============================================
// Observe Node Type - displays text results
// ============================================

NodeRegistry.register('observe', {
    label: 'Observe',
    category: 'General',
    result: true,

    ports: [
        { name: 'in', dir: 'in', type: 'any' }
    ],

    defaultConfig: {
        title: 'Observe',
        tool: '',
        menuTag: { en: 'Observer', ko: '관찰', ja: 'オブザーバー', zh: '观察器', fr: 'Observateur', de: 'Beobachter', es: 'Observador', it: 'Osservatore', pt: 'Observador', nl: 'Waarnemer', ru: 'Наблюдатель', ar: 'مراقب', hi: 'पर्यवेक्षक', tr: 'Gözlemci', pl: 'Obserwator', cs: 'Pozorovatel', sv: 'Observatör', da: 'Observatør', no: 'Observatør', fi: 'Tarkkailija', el: 'Παρατηρητής', hu: 'Megfigyelő', ro: 'Observator', uk: 'Спостерігач', vi: 'Quan sát', th: 'ผู้สังเกต', id: 'Pengamat' },
        description: { en: 'Observe and display intermediate results during execution', ko: '실행 중 중간 결과를 관찰하고 표시', ja: '実行中の中間結果を観察して表示する', zh: '在执行过程中观察和显示中间结果', fr: 'Observer et afficher les résultats intermédiaires pendant l\'exécution', de: 'Zwischenergebnisse während der Ausführung beobachten und anzeigen', es: 'Observar y mostrar resultados intermedios durante la ejecución', it: 'Osserva e visualizza i risultati intermedi durante l\'esecuzione', pt: 'Observar e exibir resultados intermediários durante a execução', nl: 'Tussenresultaten observeren en weergeven tijdens uitvoering', ru: 'Наблюдение и отображение промежуточных результатов во время выполнения', ar: 'مراقبة وعرض النتائج الوسيطة أثناء التنفيذ', hi: 'निष्पादन के दौरान मध्यवर्ती परिणामों का अवलोकन और प्रदर्शन करें', tr: 'Yürütme sırasında ara sonuçları gözlemle ve göster', pl: 'Obserwuj i wyświetlaj wyniki pośrednie podczas wykonywania', cs: 'Pozorovat a zobrazit mezivýsledky během provádění', sv: 'Observera och visa mellanresultat under körning', da: 'Observér og vis mellemresultater under udførelse', no: 'Observer og vis mellomresultater under utførelse', fi: 'Tarkkaile ja näytä välitulokset suorituksen aikana', el: 'Παρατήρηση και εμφάνιση ενδιάμεσων αποτελεσμάτων κατά την εκτέλεση', hu: 'Köztes eredmények megfigyelése és megjelenítése végrehajtás közben', ro: 'Observă și afișează rezultatele intermediare în timpul execuției', uk: 'Спостереження та відображення проміжних результатів під час виконання', vi: 'Quan sát và hiển thị kết quả trung gian trong quá trình thực thi', th: 'สังเกตและแสดงผลลัพธ์ระหว่างกลางระหว่างการดำเนินการ', id: 'Amati dan tampilkan hasil antara selama eksekusi' },
        status: 'pending',
        stepNum: '',
        resultText: ''
    },

    render(node, helpers) {
        const el = document.createElement('div');
        const result = node.resultText || '';
        const outputContent = result
            ? helpers.escapeHtml(result)
            : '<span class="ng-observe-placeholder">No result yet</span>';

        el.innerHTML = `
            <div class="ng-ports-in-row" data-node-id="${node.id}">
                <div class="ng-port ng-port-in" data-port-name="in" data-port-dir="in" data-port-type="any" data-node-id="${node.id}"></div>
            </div>
            <div class="ng-node-header">
                <span class="ng-node-num">${node.stepNum}</span>
                <span class="ng-node-title">${helpers.escapeHtml(node.title)}</span>
            </div>
            <div class="ng-node-body ng-observe-body">
                <div class="ng-observe-output">${outputContent}</div>
            </div>
        `;

        return el;
    },

    getDragHandle(el) {
        return el.querySelector('.ng-node-header');
    },

    updateResult(el, text) {
        const output = el.querySelector('.ng-observe-output');
        if (!output) return;
        if (text) {
            const div = document.createElement('div');
            div.textContent = text;
            output.innerHTML = div.innerHTML;
        } else {
            output.innerHTML = '<span class="ng-observe-placeholder">No result yet</span>';
        }
    }
});
