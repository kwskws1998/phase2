// ============================================
// Double Node Type (Input category)
// ============================================

NodeRegistry.register('double_value', {
    label: 'Double',
    category: 'Input',
    dataOnly: true,

    ports: [
        { name: 'out', dir: 'out', type: 'double', label: 'Value' }
    ],

    defaultConfig: {
        title: 'Double',
        status: 'completed',
        portValues: { out: 0.0 }
    },

    render(node, helpers) {
        const el = document.createElement('div');
        const val = (node.portValues && node.portValues.out !== undefined) ? node.portValues.out : 0.0;

        el.innerHTML = `
            <div class="ng-node-header">
                <span class="ng-node-title">${helpers.escapeHtml(node.title)}</span>
            </div>
            <div class="ng-input-node-body">
                <input class="ng-input-node-field ng-port-default ng-interactive" type="number" step="any" value="${val}"
                       data-port-ref="out">
            </div>
            <div class="ng-ports-out-row" data-node-id="${node.id}">
                <div class="ng-port ng-port-out" data-port-name="out" data-port-dir="out" data-port-type="double" data-node-id="${node.id}"></div>
            </div>
        `;

        return el;
    },

    getDragHandle(el) {
        return el.querySelector('.ng-node-header');
    }
});
