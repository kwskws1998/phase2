// ============================================
// Vector2 Node Type (Input category)
// ============================================

NodeRegistry.register('vector2', {
    label: 'Vector2',
    category: 'Input',
    dataOnly: true,

    ports: [
        { name: 'out', dir: 'out', type: 'vector2', label: 'Vector' }
    ],

    defaultConfig: {
        title: 'Vector2',
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
            en: '2D vector (x, y) input',
            ko: '2D 벡터 (x, y) 입력',
            ja: '2Dベクトル (x, y) 入力',
            zh: '2D向量 (x, y) 输入',
            fr: 'Entrée de vecteur 2D (x, y)',
            de: '2D-Vektor (x, y) Eingabe',
            es: 'Entrada de vector 2D (x, y)',
            it: 'Input vettore 2D (x, y)',
            pt: 'Entrada de vetor 2D (x, y)',
            nl: '2D-vector (x, y) invoer',
            ru: 'Ввод 2D-вектора (x, y)',
            ar: 'إدخال متجه ثنائي الأبعاد (x, y)',
            hi: '2D वेक्टर (x, y) इनपुट',
            tr: '2D vektör (x, y) girişi',
            pl: 'Wejście wektora 2D (x, y)',
            cs: 'Vstup 2D vektoru (x, y)',
            sv: '2D-vektor (x, y) inmatning',
            da: '2D-vektor (x, y) input',
            no: '2D-vektor (x, y) inngang',
            fi: '2D-vektori (x, y) syöte',
            el: 'Είσοδος 2D διανύσματος (x, y)',
            hu: '2D vektor (x, y) bemenet',
            ro: 'Intrare vector 2D (x, y)',
            uk: 'Введення 2D-вектора (x, y)',
            vi: 'Đầu vào vectơ 2D (x, y)',
            th: 'อินพุตเวกเตอร์ 2 มิติ (x, y)',
            id: 'Input vektor 2D (x, y)'
        },
        status: 'completed',
        portValues: { out: [0, 0] }
    },

    render(node, helpers) {
        const el = document.createElement('div');
        const defaults = [0, 0];
        const vals = (node.portValues && Array.isArray(node.portValues.out))
            ? node.portValues.out : defaults;

        const labels = ['X', 'Y'];
        const cells = [];
        for (let i = 0; i < 2; i++) {
            const v = vals[i] !== undefined ? vals[i] : defaults[i];
            cells.push(`<input class="ng-matrix-cell ng-interactive" type="number" step="any" value="${v}" data-cell="${i}" placeholder="${labels[i]}">`);
        }

        el.innerHTML = `
            <div class="ng-node-header">
                <span class="ng-node-title">${helpers.escapeHtml(node.title)}</span>
            </div>
            <div class="ng-matrix-body">
                <div class="ng-matrix-grid ng-vector-2">
                    ${cells.join('\n')}
                </div>
            </div>
            <div class="ng-ports-out-row" data-node-id="${node.id}">
                <div class="ng-port ng-port-out" data-port-name="out" data-port-dir="out" data-port-type="vector2" data-node-id="${node.id}"></div>
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
