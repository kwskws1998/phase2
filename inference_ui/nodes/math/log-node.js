// ============================================
// Log Node Type (Math category)
// ============================================

NodeRegistry.register('math_log', {
    label: 'Log',
    category: 'Math',

    ports: [
        { name: 'value', dir: 'in', type: 'float', label: 'Value', defaultValue: 1 },
        { name: 'base', dir: 'in', type: 'float', label: 'Base', defaultValue: 2.718 },
        { name: 'out', dir: 'out', type: 'float', label: 'Result' }
    ],

    defaultConfig: {
        title: 'Log',
        status: 'pending',
        portValues: { value: 1, base: 2.718 }
    },

    render(node, helpers) {
        return MathNodeHelper.render(node, helpers, [
            { name: 'value', label: 'Value', type: 'float', defaultValue: 1 },
            { name: 'base', label: 'Base', type: 'float', defaultValue: 2.718 }
        ]);
    },

    getDragHandle(el) {
        return el.querySelector('.ng-node-header');
    }
});
