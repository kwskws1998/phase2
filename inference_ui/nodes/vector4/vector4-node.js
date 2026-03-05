// ============================================
// Vector4 Node Type (Input category)
// ============================================

NodeRegistry.register('vector4', {
    label: 'Vector4',
    category: 'Input',
    dataOnly: true,

    ports: [
        { name: 'out', dir: 'out', type: 'vector4', label: 'Vector' }
    ],

    defaultConfig: {
        title: 'Vector4',
        menuTag: {
            en: 'Vector', ko: '벡터',
            ja: 'ベクトル', zh: '向量',
            fr: 'Vecteur', de: 'Vektor', es: 'Vector', it: 'Vettore', pt: 'Vetor',
            nl: 'Vector', ru: 'Вектор', ar: 'متجه', hi: 'वेक्टर',
            tr: 'Vektör', pl: 'Wektor', cs: 'Vektor', sv: 'Vektor',
            da: 'Vektor', no: 'Vektor', fi: 'Vektori', el: 'Διάνυσμα',
            hu: 'Vektor', ro: 'Vector', uk: 'Вектор',
            vi: 'Vectơ', th: 'เวกเตอร์', id: 'Vektor'
        },
        description: {
            en: '4D vector (x, y, z, w) input',
            ko: '4D 벡터 (x, y, z, w) 입력',
            ja: '4Dベクトル (x, y, z, w) 入力',
            zh: '4D向量 (x, y, z, w) 输入',
            fr: 'Entrée de vecteur 4D (x, y, z, w)',
            de: '4D-Vektor (x, y, z, w) Eingabe',
            es: 'Entrada de vector 4D (x, y, z, w)',
            it: 'Input vettore 4D (x, y, z, w)',
            pt: 'Entrada de vetor 4D (x, y, z, w)',
            nl: '4D-vector (x, y, z, w) invoer',
            ru: 'Ввод 4D-вектора (x, y, z, w)',
            ar: 'إدخال متجه رباعي الأبعاد (x, y, z, w)',
            hi: '4D वेक्टर (x, y, z, w) इनपुट',
            tr: '4D vektör (x, y, z, w) girişi',
            pl: 'Wejście wektora 4D (x, y, z, w)',
            cs: 'Vstup 4D vektoru (x, y, z, w)',
            sv: '4D-vektor (x, y, z, w) inmatning',
            da: '4D-vektor (x, y, z, w) input',
            no: '4D-vektor (x, y, z, w) inngang',
            fi: '4D-vektori (x, y, z, w) syöte',
            el: 'Είσοδος 4D διανύσματος (x, y, z, w)',
            hu: '4D vektor (x, y, z, w) bemenet',
            ro: 'Intrare vector 4D (x, y, z, w)',
            uk: 'Введення 4D-вектора (x, y, z, w)',
            vi: 'Đầu vào vectơ 4D (x, y, z, w)',
            th: 'อินพุตเวกเตอร์ 4 มิติ (x, y, z, w)',
            id: 'Input vektor 4D (x, y, z, w)'
        },
        status: 'completed',
        portValues: { out: [0, 0, 0, 0] }
    },

    render(node, helpers) {
        const el = document.createElement('div');
        const defaults = [0, 0, 0, 0];
        const vals = (node.portValues && Array.isArray(node.portValues.out))
            ? node.portValues.out : defaults;

        const labels = ['X', 'Y', 'Z', 'W'];
        const cells = [];
        for (let i = 0; i < 4; i++) {
            const v = vals[i] !== undefined ? vals[i] : defaults[i];
            cells.push(`<input class="ng-matrix-cell ng-interactive" type="number" step="any" value="${v}" data-cell="${i}" placeholder="${labels[i]}">`);
        }

        el.innerHTML = `
            <div class="ng-node-header">
                <span class="ng-node-title">${helpers.escapeHtml(node.title)}</span>
            </div>
            <div class="ng-matrix-body">
                <div class="ng-matrix-grid ng-vector-4">
                    ${cells.join('\n')}
                </div>
            </div>
            <div class="ng-ports-out-row" data-node-id="${node.id}">
                <div class="ng-port ng-port-out" data-port-name="out" data-port-dir="out" data-port-type="vector4" data-node-id="${node.id}"></div>
            </div>
        `;

        return el;
    },

    afterRender(el, node, helpers) {
        const updateVector = () => {
            const n = helpers.getNode();
            if (!n) return;
            if (!n.portValues) n.portValues = {};
            const cells = el.querySelectorAll('.ng-matrix-cell');
            n.portValues.out = Array.from(cells).map(c => parseFloat(c.value) || 0);
        };
        el.querySelectorAll('.ng-matrix-cell').forEach(cell => {
            cell.addEventListener('change', updateVector);
            cell.addEventListener('input', updateVector);
        });
    },

    getDragHandle(el) {
        return el.querySelector('.ng-node-header');
    }
});
