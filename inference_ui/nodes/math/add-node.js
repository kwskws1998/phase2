// ============================================
// Add Node Type (Math category)
// ============================================

NodeRegistry.register('math_add', {
    label: 'Add',
    category: 'Math',

    ports: [
        { name: 'a', dir: 'in', type: 'addable', label: 'A', defaultValue: 0 },
        { name: 'b', dir: 'in', type: 'addable', label: 'B', defaultValue: 0 },
        { name: 'out', dir: 'out', type: 'numeric', label: 'Result' }
    ],

    defaultConfig: {
        title: 'Add',
        status: 'pending',
        portValues: { a: 0, b: 0 }
    },

    render(node, helpers) {
        return MathNodeHelper.render(node, helpers, [
            { name: 'a', label: 'A', type: 'addable', defaultValue: 0 },
            { name: 'b', label: 'B', type: 'addable', defaultValue: 0 }
        ], 'numeric');
    },

    getDragHandle(el) {
        return el.querySelector('.ng-node-header');
    }
});
