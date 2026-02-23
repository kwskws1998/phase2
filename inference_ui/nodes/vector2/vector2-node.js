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
