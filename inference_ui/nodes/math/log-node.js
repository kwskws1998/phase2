// ============================================
// Log Node Type (Math category)
// ============================================

NodeRegistry.register('math_log', {
    label: 'Log',
    category: 'Math',

    ports: [
        { name: 'value', dir: 'in', type: 'numeric', label: 'Value', defaultValue: 1 },
        { name: 'base', dir: 'in', type: 'numeric', label: 'Base', defaultValue: 2.718 },
        { name: 'out', dir: 'out', type: 'numeric', label: 'Result' }
    ],

    defaultConfig: {
        title: 'Log',
        status: 'pending',
        portValues: { value: 1, base: 2.718 }
    },

    render(node, helpers) {
        return MathNodeHelper.render(node, helpers, [
            { name: 'value', label: 'Value', type: 'numeric', defaultValue: 1 },
            { name: 'base', label: 'Base', type: 'numeric', defaultValue: 2.718 }
        ], 'numeric');
    },

    getDragHandle(el) {
        return el.querySelector('.ng-node-header');
    }
});
