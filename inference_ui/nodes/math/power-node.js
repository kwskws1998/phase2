// ============================================
// Power Node Type (Math category)
// ============================================

NodeRegistry.register('math_power', {
    label: 'Power',
    category: 'Math',

    ports: [
        { name: 'base', dir: 'in', type: 'numeric', label: 'Base', defaultValue: 2 },
        { name: 'exp', dir: 'in', type: 'numeric', label: 'Exp', defaultValue: 2 },
        { name: 'out', dir: 'out', type: 'numeric', label: 'Result' }
    ],

    defaultConfig: {
        title: 'Power',
        status: 'pending',
        portValues: { base: 2, exp: 2 }
    },

    render(node, helpers) {
        return MathNodeHelper.render(node, helpers, [
            { name: 'base', label: 'Base', type: 'numeric', defaultValue: 2 },
            { name: 'exp', label: 'Exp', type: 'numeric', defaultValue: 2 }
        ], 'numeric');
    },

    getDragHandle(el) {
        return el.querySelector('.ng-node-header');
    }
});
