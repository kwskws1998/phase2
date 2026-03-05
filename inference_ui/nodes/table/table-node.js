// ============================================
// Table Node Type - displays tabular data
// ============================================

NodeRegistry.register('table', {
    label: 'Table',
    category: 'General',
    result: true,

    ports: [
        { name: 'in', dir: 'in', type: 'table' },
        { name: 'out', dir: 'out', type: 'table' }
    ],

    defaultConfig: {
        title: 'Table',
        tool: '',
        menuTag: { en: 'Table', ko: '테이블', ja: 'テーブル', zh: '表格', fr: 'Tableau', de: 'Tabelle', es: 'Tabla', it: 'Tabella', pt: 'Tabela', nl: 'Tabel', ru: 'Таблица', ar: 'جدول', hi: 'तालिका', tr: 'Tablo', pl: 'Tabela', cs: 'Tabulka', sv: 'Tabell', da: 'Tabel', no: 'Tabell', fi: 'Taulukko', el: 'Πίνακας', hu: 'Táblázat', ro: 'Tabel', uk: 'Таблиця', vi: 'Bảng', th: 'ตาราง', id: 'Tabel' },
        description: { en: 'Display structured data in an interactive table format', ko: '구조화된 데이터를 테이블 형태로 표시', ja: '構造化データをインタラクティブなテーブル形式で表示する', zh: '以交互式表格格式显示结构化数据', fr: 'Afficher des données structurées dans un format de tableau interactif', de: 'Strukturierte Daten in einem interaktiven Tabellenformat anzeigen', es: 'Mostrar datos estructurados en un formato de tabla interactivo', it: 'Visualizza dati strutturati in un formato tabellare interattivo', pt: 'Exibir dados estruturados em formato de tabela interativo', nl: 'Gestructureerde gegevens weergeven in een interactief tabelformaat', ru: 'Отображение структурированных данных в интерактивном табличном формате', ar: 'عرض البيانات المنظمة في تنسيق جدول تفاعلي', hi: 'संरचित डेटा को इंटरैक्टिव तालिका प्रारूप में प्रदर्शित करें', tr: 'Yapılandırılmış verileri etkileşimli tablo biçiminde göster', pl: 'Wyświetl dane strukturalne w interaktywnym formacie tabeli', cs: 'Zobrazit strukturovaná data v interaktivním formátu tabulky', sv: 'Visa strukturerad data i ett interaktivt tabellformat', da: 'Vis strukturerede data i et interaktivt tabelformat', no: 'Vis strukturerte data i et interaktivt tabellformat', fi: 'Näytä rakenteellinen data interaktiivisessa taulukkomuodossa', el: 'Εμφάνιση δομημένων δεδομένων σε διαδραστική μορφή πίνακα', hu: 'Strukturált adatok megjelenítése interaktív táblázat formátumban', ro: 'Afișează date structurate într-un format de tabel interactiv', uk: 'Відображення структурованих даних в інтерактивному табличному форматі', vi: 'Hiển thị dữ liệu có cấu trúc ở định dạng bảng tương tác', th: 'แสดงข้อมูลที่มีโครงสร้างในรูปแบบตารางแบบโต้ตอบ', id: 'Tampilkan data terstruktur dalam format tabel interaktif' },
        status: 'pending',
        stepNum: '',
        resultText: ''
    },

    render(node, helpers) {
        const el = document.createElement('div');
        const hasResult = node.resultText && node.resultText.length > 0;

        el.innerHTML = `
            <div class="ng-ports-in-row" data-node-id="${node.id}">
                <div class="ng-port ng-port-in" data-port-name="in" data-port-dir="in" data-port-type="table" data-node-id="${node.id}"></div>
            </div>
            <div class="ng-node-header">
                <span class="ng-node-num">${node.stepNum}</span>
                <span class="ng-node-title">${helpers.escapeHtml(node.title)}</span>
            </div>
            <div class="ng-node-body ng-table-body">
                <div class="ng-table-output">
                    <span class="ng-table-placeholder">No data</span>
                </div>
            </div>
            <div class="ng-node-progress"></div>
            <div class="ng-ports-out-row" data-node-id="${node.id}">
                <div class="ng-port ng-port-out" data-port-name="out" data-port-dir="out" data-port-type="table" data-node-id="${node.id}"></div>
            </div>
        `;

        if (hasResult) {
            const output = el.querySelector('.ng-table-output');
            this._renderTableData(output, node.resultText);
        }

        return el;
    },

    getDragHandle(el) {
        return el.querySelector('.ng-node-header');
    },

    _renderTableData(container, data) {
        if (!container || !data) return;

        let rows;
        if (typeof data === 'string') {
            try { rows = JSON.parse(data); } catch { rows = null; }
        } else if (Array.isArray(data)) {
            rows = data;
        }

        if (!Array.isArray(rows) || rows.length === 0) {
            container.textContent = typeof data === 'string' ? data : JSON.stringify(data);
            return;
        }

        const escapeCell = v => {
            const d = document.createElement('span');
            d.textContent = v == null ? '' : String(v);
            return d.innerHTML;
        };

        const headerRow = rows[0];
        let html = '<table><thead><tr>';
        for (const h of headerRow) html += `<th>${escapeCell(h)}</th>`;
        html += '</tr></thead><tbody>';
        for (let r = 1; r < rows.length; r++) {
            html += '<tr>';
            for (const c of rows[r]) html += `<td>${escapeCell(c)}</td>`;
            html += '</tr>';
        }
        html += '</tbody></table>';
        container.innerHTML = html;
    },

    updateResult(el, data) {
        const output = el.querySelector('.ng-table-output');
        if (!output) return;
        if (data) {
            this._renderTableData(output, data);
        } else {
            output.innerHTML = '<span class="ng-table-placeholder">No data</span>';
        }
    }
});
