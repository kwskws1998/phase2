// ============================================
// Color Node Type (Input category)
// ============================================

NodeRegistry.register('color_value', {
    label: 'Color',
    category: 'Input',
    dataOnly: true,

    ports: [
        { name: 'out', dir: 'out', type: 'color', label: 'Color' }
    ],

    defaultConfig: {
        title: 'Color',
        menuTag: {
            en: 'Color', ko: '색상',
            ja: '色', zh: '颜色',
            fr: 'Couleur', de: 'Farbe', es: 'Color', it: 'Colore', pt: 'Cor',
            nl: 'Kleur', ru: 'Цвет', ar: 'لون', hi: 'रंग',
            tr: 'Renk', pl: 'Kolor', cs: 'Barva', sv: 'Färg',
            da: 'Farve', no: 'Farge', fi: 'Väri', el: 'Χρώμα',
            hu: 'Szín', ro: 'Culoare', uk: 'Колір',
            vi: 'Màu sắc', th: 'สี', id: 'Warna'
        },
        description: {
            en: 'RGBA color picker input',
            ko: 'RGBA 색상 선택기',
            ja: 'RGBAカラーピッカー入力',
            zh: 'RGBA颜色选择器输入',
            fr: 'Sélecteur de couleur RGBA',
            de: 'RGBA-Farbauswahl-Eingabe',
            es: 'Selector de color RGBA',
            it: 'Selettore di colore RGBA',
            pt: 'Seletor de cor RGBA',
            nl: 'RGBA-kleurkiezer invoer',
            ru: 'Ввод цвета RGBA',
            ar: 'منتقي ألوان RGBA',
            hi: 'RGBA रंग चयनकर्ता इनपुट',
            tr: 'RGBA renk seçici girişi',
            pl: 'Selektor kolorów RGBA',
            cs: 'Vstup výběru barvy RGBA',
            sv: 'RGBA-färgväljarinmatning',
            da: 'RGBA-farvevælger-input',
            no: 'RGBA-fargevelgerinngang',
            fi: 'RGBA-värinvalitsinsyöte',
            el: 'Είσοδος επιλογής χρώματος RGBA',
            hu: 'RGBA színválasztó bemenet',
            ro: 'Selector de culoare RGBA',
            uk: 'Введення вибору кольору RGBA',
            vi: 'Bộ chọn màu RGBA',
            th: 'ตัวเลือกสี RGBA',
            id: 'Pemilih warna RGBA'
        },
        status: 'completed',
        portValues: { out: [255, 255, 255, 1.0] }
    },

    render(node, helpers) {
        const el = document.createElement('div');
        const defaults = [255, 255, 255, 1.0];
        const vals = (node.portValues && Array.isArray(node.portValues.out))
            ? node.portValues.out : defaults;

        const r = vals[0] !== undefined ? vals[0] : 255;
        const g = vals[1] !== undefined ? vals[1] : 255;
        const b = vals[2] !== undefined ? vals[2] : 255;
        const a = vals[3] !== undefined ? vals[3] : 1.0;

        const hex = '#' + [r, g, b].map(c => Math.max(0, Math.min(255, Math.round(c))).toString(16).padStart(2, '0')).join('');

        el.innerHTML = `
            <div class="ng-node-header">
                <span class="ng-node-title">${helpers.escapeHtml(node.title)}</span>
            </div>
            <div class="ng-color-body">
                <input class="ng-color-picker ng-interactive" type="color" value="${hex}">
                <div class="ng-color-fields">
                    <input class="ng-matrix-cell ng-interactive" type="number" min="0" max="255" step="1" value="${r}" data-channel="0" placeholder="R">
                    <input class="ng-matrix-cell ng-interactive" type="number" min="0" max="255" step="1" value="${g}" data-channel="1" placeholder="G">
                    <input class="ng-matrix-cell ng-interactive" type="number" min="0" max="255" step="1" value="${b}" data-channel="2" placeholder="B">
                    <input class="ng-matrix-cell ng-interactive" type="number" min="0" max="1" step="0.01" value="${a}" data-channel="3" placeholder="A">
                </div>
            </div>
            <div class="ng-ports-out-row" data-node-id="${node.id}">
                <div class="ng-port ng-port-out" data-port-name="out" data-port-dir="out" data-port-type="color" data-node-id="${node.id}"></div>
            </div>
        `;

        return el;
    },

    afterRender(el, node, helpers) {
        const picker = el.querySelector('.ng-color-picker');
        const fields = el.querySelectorAll('.ng-color-fields .ng-matrix-cell');
        if (!picker || fields.length < 4) return;

        const syncToNode = () => {
            const n = helpers.getNode();
            if (!n) return;
            if (!n.portValues) n.portValues = {};
            n.portValues.out = [
                parseInt(fields[0].value) || 0,
                parseInt(fields[1].value) || 0,
                parseInt(fields[2].value) || 0,
                parseFloat(fields[3].value) || 0
            ];
        };

        picker.addEventListener('input', () => {
            const hex = picker.value;
            fields[0].value = parseInt(hex.slice(1, 3), 16);
            fields[1].value = parseInt(hex.slice(3, 5), 16);
            fields[2].value = parseInt(hex.slice(5, 7), 16);
            syncToNode();
        });

        const syncPickerFromFields = () => {
            const r = Math.max(0, Math.min(255, parseInt(fields[0].value) || 0));
            const g = Math.max(0, Math.min(255, parseInt(fields[1].value) || 0));
            const b = Math.max(0, Math.min(255, parseInt(fields[2].value) || 0));
            picker.value = '#' + [r, g, b].map(c => c.toString(16).padStart(2, '0')).join('');
            syncToNode();
        };

        fields.forEach(f => {
            f.addEventListener('change', syncPickerFromFields);
            f.addEventListener('input', syncPickerFromFields);
        });
    },

    getDragHandle(el) {
        return el.querySelector('.ng-node-header');
    }
});
