
import { JsonPatchOperation } from '../types';

export function applyPatch(doc: any, patch: JsonPatchOperation[]): any {
  patch.forEach(op => {
    const path = op.path.split('/').slice(1);
    let target = doc;

    for (let i = 0; i < path.length - 1; i++) {
      target = target[path[i]];
      if (target === undefined) {
        throw new Error(`Invalid path: ${op.path}`);
      }
    }

    const key = path[path.length - 1];

    switch (op.op) {
      case 'add':
      case 'replace':
        if (key === 'tasks' && op.path === '/tasks') {
            // Special case for replacing the entire tasks object
            doc.tasks = op.value;
        } else {
            target[key] = op.value;
        }
        break;
      case 'remove':
        if (Array.isArray(target)) {
          target.splice(parseInt(key, 10), 1);
        } else {
          delete target[key];
        }
        break;
      default:
        throw new Error(`Unsupported operation: ${op.op}`);
    }
  });
  return doc;
}
