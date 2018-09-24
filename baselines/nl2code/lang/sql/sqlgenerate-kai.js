/**
 * Created by kai on 2018/4/23.
 */
import 'babel-polyfill';
import {map, join, head, compose, curry, toUpper, prop, equals, isEmpty, F, isArrayLike, concat, __, pluck, contains} from 'ramda';

const INDENT = '\t';
const LINE_END = '\n';

// This allows calling a function recursivly based on node type. Some of the
// nodes have non-standard types and so we need to modify how we call the
// Generator.
const recurse = curry((Generator, n) => {
    switch (n.type) {
        case 'function':
            return Generator['function'](n);
        case 'module':
            return Generator.module(n);
        case 'assignment':
            return Generator.assignment(n);
        case 'event':
            return Generator.event(n);
        default:
            return Generator[n.type][n.variant](n);
    }
});

const mapr = compose(map, recurse);

const datatype = (n) => n.variant;
const returnNewLine = join('\n');
const joinList = join(', ');
const terminateStatements = map(concat(__, ';'));
const containsSelect = (s) => (s.indexOf('SELECT') !== -1);
const isOfFormat = (n) => compose(equals(n), prop('format'));

var Generator = {
    assignment : (n) => {
        const recurser = recurse(Generator);
        const target = recurser(n.target);
        const value = recurser(n.value);
        return `${target} = ${value}`;
    },
    statement : (n) => {
            const recurser = recurse(Generator);
            const recourseList = mapr(Generator);
            const argsList = compose(joinList, recourseList);

            var str = [''];
            if (n.with) {
                const withS = recourseList(n.with);
                const isRecursive = (n) => isArrayLike(n) ? compose(contains('recursive'), pluck('variant'))(n) : F;
                const w = isRecursive(n.with) ? 'WITH RECURSIVE' : 'WITH';
                str.push(`${w} ${withS}${LINE_END}`);
            }
            str.push('SELECT ');
            if (n.result) {
                const results = argsList(n.result);
                str.push(`${results}${LINE_END}`);
            }
            if (n.from) {
                const from = recurser(n.from);
                str.push(`${INDENT}FROM ${from}${LINE_END}`);
            }
            if (n.where) {
                const where = recurser(head(n.where));
                str.push(`${INDENT}WHERE ${where}${LINE_END}`);
            }
            if (n.group) {
                const group = recurser(n.group);
                str.push(`${INDENT}GROUP BY ${group}${LINE_END}`);
            }
            if (n.having) {
                const having = recurser(n.having);
                str.push(`${INDENT}HAVING ${having}${LINE_END}`);
            }
            if (n.order) {
                const order = recourseList(n.order);
                str.push(`${INDENT}ORDER BY ${order}${LINE_END}`);
            }
            if (n.limit) {
                const limit = recurser(n.limit);
                str.push(`${INDENT}${limit}`);
            }
            return str.join('');
        },
    // compound : {
    //     union : (n) => {
    //         const statement = recurse(Generator)(n.statement);
    //         return `${toUpper(n.variant)}${LINE_END}${statement}`;
    //     },
    //     get 'union all'(){
    //         return this.union;
    //     },
    //     get 'except'(){
    //         return this.union;
    //     },
    //     get 'intersect'(){
    //         return this.union;
    //     },
    // },
    identifier : {
        star : (n) => n.name,
        table : (n) => {
            const alias =  (n.alias)  ? `AS ${n.alias}` : '';
            const index = (n.index) ? recurse(Generator)(n.index) : '';
            return `${n.name} ${alias} ${index}`;
        },
        column : (n) => {
            const recurser = recurse(Generator);
            const alias =  (n.alias) ? `AS ${n.alias}` : '';
            const index = (n.index) ? recurser(n.index) : '';
            return `${n.name} ${alias} ${index}`;
        },
        'function' : (n) => n.name,
        expression : (n) => {
            const m = mapr(Generator);
            return `${n.name}(${m(n.columns)})`;
        }
    },
    literal : {
        text : (n) => `'${n.value}'`,
        decimal : (n) => `${n.value}`,
        null : () => 'NULL'
    },
    expression : {
        operation : (n) => {
            const recurser = recurse(Generator);
            const isUnaryOperation = isOfFormat('unary');

            if(isUnaryOperation(n)){
                const expression = recurser(n.expression);
                const operator = (n.operator) ? `${n.operator}` : '';
                const alias = (n.alias) ? `AS [${n.alias}]` : '';
                return `${operator} ${expression} ${alias}`;
            }


            const isBetween = (n) => (n.operation === 'between');
            const isExpression = (n) => (n.type === 'expression');

            const side = (s) => {
                const sideOp = recurser(n[s]);
                return !isBetween(n) && (isExpression(n[s]) || containsSelect(sideOp)) ? `(${sideOp})` : sideOp;
            };
            const left = side('left');
            const right = side('right');

            return `${left} ${n.operation} ${right}`;
        },
        list : (n) => {
            const argsList = compose(joinList, mapr(Generator));
            return argsList(n.expression);
        },
        order : (n) => {
            const recurser = recurse(Generator);
            const expression = recurser(n.expression);
            const direction = n.direction;
            return `${expression} ${toUpper(direction)}`;
        },
        limit : (n) => {
            const recurser = recurse(Generator);
            const limit = recurser(n.start);
            const offset = n.offset ? `OFFSET ${recurser(n.offset)}` : '';
            return `LIMIT ${limit}${LINE_END}${INDENT}${offset}`;
        },
        exists : (n) => n.operator
    },
    'function' : (n) => {
        const recurser = recurse(Generator);
        const name = toUpper(recurser(n.name));
        const args = recurser(n.args);
        const alias =  (n.alias)  ? `AS ${n.alias}` : '';
        return `${name}(${args}) ${alias}`;
    },
    map : {
        join : (n) => {
            const recurser = recurse(Generator);
            const source = recurser(n.source);
            const sourceAlias = (n.source.alias)? n.source.alias : '';
            const join = recurser(head(n.map));

            // Its a select subquery
            if (containsSelect(source)){
                const subquery = `(${source}) AS ${sourceAlias} ${join}`;
                return subquery;
            }
            // Its an inner join.
            return `${source} ${join}`;
        }
    },
    join : {
        join : (n) => {
            const recurser = recurse(Generator);
            const source = recurser(n.source);
            const constraint = recurser(n.constraint);
            return `${INDENT}JOIN ${source}${LINE_END}${constraint}`;
        },
        'inner join' : (n) => {
            const recurser = recurse(Generator);
            const source = recurser(n.source);
            const sourceAlias = (n.source.alias)? ` AS ${n.source.alias}` : '';
            const constraint = recurser(n.constraint);
            return `${INDENT}INNER JOIN (${source})${sourceAlias}${LINE_END}${constraint}`;
        },
        'left outer join' : (n) => {
            const recurser = recurse(Generator);
            const source = recurser(n.source);
            const constraint = recurser(n.constraint);
            return `${INDENT}LEFT OUTER JOIN ${source}${LINE_END}${constraint}`;
        },
        'cross join' : (n) => {
            const recurser = recurse(Generator);
            const source = recurser(n.source);
            return `, ${source}`;
        }
    }
};

module.exports = {
    generate        : (n) => Generator[n.type][n.variant](n)
};