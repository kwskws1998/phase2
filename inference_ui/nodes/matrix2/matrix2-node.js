// ============================================
// Matrix 2x2 Node Type (Input category)
// ============================================

NodeRegistry.register('matrix2', {
    label: 'Matrix 2x2',
    category: 'Input',
    dataOnly: true,

    ports: [
        { name: 'out', dir: 'out', type: 'matrix', label: 'Matrix' }
    ],

    defaultConfig: {
        title: 'Matrix 2x2',
        menuTag: { en: 'Matrix', ko: '행렬', ja: '行列', zh: '矩阵', fr: 'Matrice', de: 'Matrix', es: 'Matriz', it: 'Matrice', pt: 'Matriz', nl: 'Matrix', ru: 'Матрица', ar: 'مصفوفة', hi: 'मैट्रिक्स', tr: 'Matris', pl: 'Macierz', cs: 'Matice', sv: 'Matris', da: 'Matrix', no: 'Matrise', fi: 'Matriisi', el: 'Μήτρα', hu: 'Mátrix', ro: 'Matrice', uk: 'Матриця', vi: 'Ma trận', th: 'เมทริกซ์', id: 'Matriks' },
        description: { en: '2x2 matrix input', ko: '2x2 행렬 입력', ja: '2x2 行列入力', zh: '2x2 矩阵输入', fr: 'Entrée matrice 2x2', de: '2x2-Matrixeingabe', es: 'Entrada de matriz 2x2', it: 'Input matrice 2x2', pt: 'Entrada de matriz 2x2', nl: '2x2 matrix-invoer', ru: 'Ввод матрицы 2x2', ar: 'إدخال مصفوفة 2x2', hi: '2x2 मैट्रिक्स इनपुट', tr: '2x2 matris girişi', pl: 'Wejście macierzy 2x2', cs: 'Vstup matice 2x2', sv: '2x2 matrisinmatning', da: '2x2 matrix-input', no: '2x2 matriseinndata', fi: '2x2-matriisin syöttö', el: 'Είσοδος μήτρας 2x2', hu: '2x2 mátrix bevitel', ro: 'Intrare matrice 2x2', uk: 'Введення матриці 2x2', vi: 'Nhập ma trận 2x2', th: 'อินพุตเมทริกซ์ 2x2', id: 'Input matriks 2x2' },
        status: 'completed',
        portValues: { out: [1,0, 0,1] }
    },

    render(node, helpers) {
        const el = document.createElement('div');
        const identity = [1,0, 0,1];
        const vals = (node.portValues && Array.isArray(node.portValues.out))
            ? node.portValues.out : identity;

        const cells = [];
        for (let i = 0; i < 4; i++) {
            const v = vals[i] !== undefined ? vals[i] : identity[i];
            cells.push(`<input class="ng-matrix-cell ng-interactive" type="number" step="any" value="${v}" data-cell="${i}">`);
        }

        el.innerHTML = `
            <div class="ng-node-header">
                <span class="ng-node-title">${helpers.escapeHtml(node.title)}</span>
            </div>
            <div class="ng-matrix-body">
                <div class="ng-matrix-grid ng-matrix-2x2">
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
