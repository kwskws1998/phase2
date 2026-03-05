// ============================================
// Vector3 Node Type (Input category)
// ============================================

NodeRegistry.register('vector3', {
    label: 'Vector3',
    category: 'Input',
    dataOnly: true,

    ports: [
        { name: 'out', dir: 'out', type: 'vector3', label: 'Vector' }
    ],

    defaultConfig: {
        title: 'Vector3',
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
            en: '3D vector (x, y, z) input',
            ko: '3D 벡터 (x, y, z) 입력',
            ja: '3Dベクトル (x, y, z) 入力',
            zh: '3D向量 (x, y, z) 输入',
            fr: 'Entrée de vecteur 3D (x, y, z)',
            de: '3D-Vektor (x, y, z) Eingabe',
            es: 'Entrada de vector 3D (x, y, z)',
            it: 'Input vettore 3D (x, y, z)',
            pt: 'Entrada de vetor 3D (x, y, z)',
            nl: '3D-vector (x, y, z) invoer',
            ru: 'Ввод 3D-вектора (x, y, z)',
            ar: 'إدخال متجه ثلاثي الأبعاد (x, y, z)',
            hi: '3D वेक्टर (x, y, z) इनपुट',
            tr: '3D vektör (x, y, z) girişi',
            pl: 'Wejście wektora 3D (x, y, z)',
            cs: 'Vstup 3D vektoru (x, y, z)',
            sv: '3D-vektor (x, y, z) inmatning',
            da: '3D-vektor (x, y, z) input',
            no: '3D-vektor (x, y, z) inngang',
            fi: '3D-vektori (x, y, z) syöte',
            el: 'Είσοδος 3D διανύσματος (x, y, z)',
            hu: '3D vektor (x, y, z) bemenet',
            ro: 'Intrare vector 3D (x, y, z)',
            uk: 'Введення 3D-вектора (x, y, z)',
            vi: 'Đầu vào vectơ 3D (x, y, z)',
            th: 'อินพุตเวกเตอร์ 3 มิติ (x, y, z)',
            id: 'Input vektor 3D (x, y, z)'
        },
        status: 'completed',
        portValues: { out: [0, 0, 0] }
    },

    render(node, helpers) {
        const el = document.createElement('div');
        const defaults = [0, 0, 0];
        const vals = (node.portValues && Array.isArray(node.portValues.out))
            ? node.portValues.out : defaults;

        const labels = ['X', 'Y', 'Z'];
        const cells = [];
        for (let i = 0; i < 3; i++) {
            const v = vals[i] !== undefined ? vals[i] : defaults[i];
            cells.push(`<input class="ng-matrix-cell ng-interactive" type="number" step="any" value="${v}" data-cell="${i}" placeholder="${labels[i]}">`);
        }

        el.innerHTML = `
            <div class="ng-node-header">
                <span class="ng-node-title">${helpers.escapeHtml(node.title)}</span>
            </div>
            <div class="ng-matrix-body">
                <div class="ng-matrix-grid ng-vector-3">
                    ${cells.join('\n')}
                </div>
            </div>
            <div class="ng-ports-out-row" data-node-id="${node.id}">
                <div class="ng-port ng-port-out" data-port-name="out" data-port-dir="out" data-port-type="vector3" data-node-id="${node.id}"></div>
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
