// ============================================
// Power Node Type (Math category)
// ============================================

NodeRegistry.register('math_power', {
    label: 'Power',
    category: 'Math',

    ports: [
        { name: 'base', dir: 'in', type: 'float', label: 'Base', defaultValue: 2 },
        { name: 'exp', dir: 'in', type: 'float', label: 'Exp', defaultValue: 2 },
        { name: 'out', dir: 'out', type: 'float', label: 'Result' }
    ],

    defaultConfig: {
        title: 'Power',
        status: 'pending',
        portValues: { base: 2, exp: 2 }
    },

    render(node, helpers) {
        return MathNodeHelper.render(node, helpers, [
            { name: 'base', label: 'Base', type: 'float', defaultValue: 2 },
            { name: 'exp', label: 'Exp', type: 'float', defaultValue: 2 }
        ]);
    },

    getDragHandle(el) {
        return el.querySelector('.ng-node-header');
    }
});
