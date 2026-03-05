// ============================================
// Matrix 3x3 Node Type (Input category)
// ============================================

NodeRegistry.register('matrix3', {
    label: 'Matrix 3x3',
    category: 'Input',
    dataOnly: true,

    ports: [
        { name: 'out', dir: 'out', type: 'matrix', label: 'Matrix' }
    ],

    defaultConfig: {
        title: 'Matrix 3x3',
        menuTag: { en: 'Matrix', ko: '행렬', ja: '行列', zh: '矩阵', fr: 'Matrice', de: 'Matrix', es: 'Matriz', it: 'Matrice', pt: 'Matriz', nl: 'Matrix', ru: 'Матрица', ar: 'مصفوفة', hi: 'मैट्रिक्स', tr: 'Matris', pl: 'Macierz', cs: 'Matice', sv: 'Matris', da: 'Matrix', no: 'Matrise', fi: 'Matriisi', el: 'Μήτρα', hu: 'Mátrix', ro: 'Matrice', uk: 'Матриця', vi: 'Ma trận', th: 'เมทริกซ์', id: 'Matriks' },
        description: { en: '3x3 matrix input', ko: '3x3 행렬 입력', ja: '3x3 行列入力', zh: '3x3 矩阵输入', fr: 'Entrée matrice 3x3', de: '3x3-Matrixeingabe', es: 'Entrada de matriz 3x3', it: 'Input matrice 3x3', pt: 'Entrada de matriz 3x3', nl: '3x3 matrix-invoer', ru: 'Ввод матрицы 3x3', ar: 'إدخال مصفوفة 3x3', hi: '3x3 मैट्रिक्स इनपुट', tr: '3x3 matris girişi', pl: 'Wejście macierzy 3x3', cs: 'Vstup matice 3x3', sv: '3x3 matrisinmatning', da: '3x3 matrix-input', no: '3x3 matriseinndata', fi: '3x3-matriisin syöttö', el: 'Είσοδος μήτρας 3x3', hu: '3x3 mátrix bevitel', ro: 'Intrare matrice 3x3', uk: 'Введення матриці 3x3', vi: 'Nhập ma trận 3x3', th: 'อินพุตเมทริกซ์ 3x3', id: 'Input matriks 3x3' },
        status: 'completed',
        portValues: { out: [1,0,0, 0,1,0, 0,0,1] }
    },

    render(node, helpers) {
        const el = document.createElement('div');
        const identity = [1,0,0, 0,1,0, 0,0,1];
        const vals = (node.portValues && Array.isArray(node.portValues.out))
            ? node.portValues.out : identity;

        const cells = [];
        for (let i = 0; i < 9; i++) {
            const v = vals[i] !== undefined ? vals[i] : identity[i];
            cells.push(`<input class="ng-matrix-cell ng-interactive" type="number" step="any" value="${v}" data-cell="${i}">`);
        }

        el.innerHTML = `
            <div class="ng-node-header">
                <span class="ng-node-title">${helpers.escapeHtml(node.title)}</span>
            </div>
            <div class="ng-matrix-body">
                <div class="ng-matrix-grid ng-matrix-3x3">
                    ${cells.join('\n')}
                </div>
            </div>
            <div class="ng-ports-out-row" data-node-id="${node.id}">
                <div class="ng-port ng-port-out" data-port-name="out" data-port-dir="out" data-port-type="matrix" data-node-id="${node.id}"></div>
            </div>
        `;

        return el;
    },

    afterRender(el, node, helpers) {
        const updateMatrix = () => {
            const n = helpers.getNode();
            if (!n) return;
            if (!n.portValues) n.portValues = {};
            const cells = el.querySelectorAll('.ng-matrix-cell');
            n.portValues.out = Array.from(cells).map(c => parseFloat(c.value) || 0);
        };
        el.querySelectorAll('.ng-matrix-cell').forEach(cell => {
            cell.addEventListener('change', updateMatrix);
            cell.addEventListener('input', updateMatrix);
        });
    },

    getDragHandle(el) {
        return el.querySelector('.ng-node-header');
    }
});
