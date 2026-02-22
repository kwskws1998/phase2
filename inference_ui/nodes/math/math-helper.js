// ============================================
// Shared render helper for Math nodes
// ============================================

const MathNodeHelper = {
    /**
     * Renders a math node with typed input ports, default value fields, and an output port.
     * @param {Object} node - The node data object
     * @param {Object} helpers - { escapeHtml }
     * @param {Array} inputDefs - [{ name, label, type, defaultValue }]
     * @returns {HTMLElement}
     */
    render(node, helpers, inputDefs) {
        const el = document.createElement('div');
        const portValues = node.portValues || {};

        const inPortsHtml = inputDefs.map(def =>
            `<div class="ng-port ng-port-in" data-port-name="${def.name}" data-port-dir="in" data-port-type="${def.type}" data-node-id="${node.id}"></div>`
        ).join('\n');

        const fieldsHtml = inputDefs.map(def => {
            const val = portValues[def.name] !== undefined ? portValues[def.name] : def.defaultValue;
            const dotColor = `var(--port-${def.type})`;
            return `<div class="ng-port-field" data-port-ref="${def.name}">
                <span class="ng-port-dot" style="background: ${dotColor}"></span>
                <span class="ng-port-label">${def.label}</span>
                <input class="ng-port-default ng-interactive" type="number" step="any" value="${val}">
            </div>`;
        }).join('\n');

        el.innerHTML = `
            <div class="ng-ports-in-row" data-node-id="${node.id}">
                ${inPortsHtml}
            </div>
            <div class="ng-node-header">
                <span class="ng-node-title">${helpers.escapeHtml(node.title)}</span>
            </div>
            <div class="ng-node-body ng-math-body">
                ${fieldsHtml}
            </div>
            <div class="ng-node-progress"></div>
            <div class="ng-ports-out-row" data-node-id="${node.id}">
                <div class="ng-port ng-port-out" data-port-name="out" data-port-dir="out" data-port-type="float" data-node-id="${node.id}"></div>
            </div>
        `;

        return el;
    }
};
