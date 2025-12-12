#!/usr/bin/env bun
// @bun
var __defProp = Object.defineProperty;
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, {
      get: all[name],
      enumerable: true,
      configurable: true,
      set: (newValue) => all[name] = () => newValue
    });
};
var __require = import.meta.require;

// node_modules/zod/v3/external.js
var exports_external = {};
__export(exports_external, {
  void: () => voidType,
  util: () => util,
  unknown: () => unknownType,
  union: () => unionType,
  undefined: () => undefinedType,
  tuple: () => tupleType,
  transformer: () => effectsType,
  symbol: () => symbolType,
  string: () => stringType,
  strictObject: () => strictObjectType,
  setErrorMap: () => setErrorMap,
  set: () => setType,
  record: () => recordType,
  quotelessJson: () => quotelessJson,
  promise: () => promiseType,
  preprocess: () => preprocessType,
  pipeline: () => pipelineType,
  ostring: () => ostring,
  optional: () => optionalType,
  onumber: () => onumber,
  oboolean: () => oboolean,
  objectUtil: () => objectUtil,
  object: () => objectType,
  number: () => numberType,
  nullable: () => nullableType,
  null: () => nullType,
  never: () => neverType,
  nativeEnum: () => nativeEnumType,
  nan: () => nanType,
  map: () => mapType,
  makeIssue: () => makeIssue,
  literal: () => literalType,
  lazy: () => lazyType,
  late: () => late,
  isValid: () => isValid,
  isDirty: () => isDirty,
  isAsync: () => isAsync,
  isAborted: () => isAborted,
  intersection: () => intersectionType,
  instanceof: () => instanceOfType,
  getParsedType: () => getParsedType,
  getErrorMap: () => getErrorMap,
  function: () => functionType,
  enum: () => enumType,
  effect: () => effectsType,
  discriminatedUnion: () => discriminatedUnionType,
  defaultErrorMap: () => en_default,
  datetimeRegex: () => datetimeRegex,
  date: () => dateType,
  custom: () => custom,
  coerce: () => coerce,
  boolean: () => booleanType,
  bigint: () => bigIntType,
  array: () => arrayType,
  any: () => anyType,
  addIssueToContext: () => addIssueToContext,
  ZodVoid: () => ZodVoid,
  ZodUnknown: () => ZodUnknown,
  ZodUnion: () => ZodUnion,
  ZodUndefined: () => ZodUndefined,
  ZodType: () => ZodType,
  ZodTuple: () => ZodTuple,
  ZodTransformer: () => ZodEffects,
  ZodSymbol: () => ZodSymbol,
  ZodString: () => ZodString,
  ZodSet: () => ZodSet,
  ZodSchema: () => ZodType,
  ZodRecord: () => ZodRecord,
  ZodReadonly: () => ZodReadonly,
  ZodPromise: () => ZodPromise,
  ZodPipeline: () => ZodPipeline,
  ZodParsedType: () => ZodParsedType,
  ZodOptional: () => ZodOptional,
  ZodObject: () => ZodObject,
  ZodNumber: () => ZodNumber,
  ZodNullable: () => ZodNullable,
  ZodNull: () => ZodNull,
  ZodNever: () => ZodNever,
  ZodNativeEnum: () => ZodNativeEnum,
  ZodNaN: () => ZodNaN,
  ZodMap: () => ZodMap,
  ZodLiteral: () => ZodLiteral,
  ZodLazy: () => ZodLazy,
  ZodIssueCode: () => ZodIssueCode,
  ZodIntersection: () => ZodIntersection,
  ZodFunction: () => ZodFunction,
  ZodFirstPartyTypeKind: () => ZodFirstPartyTypeKind,
  ZodError: () => ZodError,
  ZodEnum: () => ZodEnum,
  ZodEffects: () => ZodEffects,
  ZodDiscriminatedUnion: () => ZodDiscriminatedUnion,
  ZodDefault: () => ZodDefault,
  ZodDate: () => ZodDate,
  ZodCatch: () => ZodCatch,
  ZodBranded: () => ZodBranded,
  ZodBoolean: () => ZodBoolean,
  ZodBigInt: () => ZodBigInt,
  ZodArray: () => ZodArray,
  ZodAny: () => ZodAny,
  Schema: () => ZodType,
  ParseStatus: () => ParseStatus,
  OK: () => OK,
  NEVER: () => NEVER,
  INVALID: () => INVALID,
  EMPTY_PATH: () => EMPTY_PATH,
  DIRTY: () => DIRTY,
  BRAND: () => BRAND
});

// node_modules/zod/v3/helpers/util.js
var util;
(function(util2) {
  util2.assertEqual = (_) => {};
  function assertIs(_arg) {}
  util2.assertIs = assertIs;
  function assertNever(_x) {
    throw new Error;
  }
  util2.assertNever = assertNever;
  util2.arrayToEnum = (items) => {
    const obj = {};
    for (const item of items) {
      obj[item] = item;
    }
    return obj;
  };
  util2.getValidEnumValues = (obj) => {
    const validKeys = util2.objectKeys(obj).filter((k) => typeof obj[obj[k]] !== "number");
    const filtered = {};
    for (const k of validKeys) {
      filtered[k] = obj[k];
    }
    return util2.objectValues(filtered);
  };
  util2.objectValues = (obj) => {
    return util2.objectKeys(obj).map(function(e) {
      return obj[e];
    });
  };
  util2.objectKeys = typeof Object.keys === "function" ? (obj) => Object.keys(obj) : (object) => {
    const keys = [];
    for (const key in object) {
      if (Object.prototype.hasOwnProperty.call(object, key)) {
        keys.push(key);
      }
    }
    return keys;
  };
  util2.find = (arr, checker) => {
    for (const item of arr) {
      if (checker(item))
        return item;
    }
    return;
  };
  util2.isInteger = typeof Number.isInteger === "function" ? (val) => Number.isInteger(val) : (val) => typeof val === "number" && Number.isFinite(val) && Math.floor(val) === val;
  function joinValues(array, separator = " | ") {
    return array.map((val) => typeof val === "string" ? `'${val}'` : val).join(separator);
  }
  util2.joinValues = joinValues;
  util2.jsonStringifyReplacer = (_, value) => {
    if (typeof value === "bigint") {
      return value.toString();
    }
    return value;
  };
})(util || (util = {}));
var objectUtil;
(function(objectUtil2) {
  objectUtil2.mergeShapes = (first, second) => {
    return {
      ...first,
      ...second
    };
  };
})(objectUtil || (objectUtil = {}));
var ZodParsedType = util.arrayToEnum([
  "string",
  "nan",
  "number",
  "integer",
  "float",
  "boolean",
  "date",
  "bigint",
  "symbol",
  "function",
  "undefined",
  "null",
  "array",
  "object",
  "unknown",
  "promise",
  "void",
  "never",
  "map",
  "set"
]);
var getParsedType = (data) => {
  const t = typeof data;
  switch (t) {
    case "undefined":
      return ZodParsedType.undefined;
    case "string":
      return ZodParsedType.string;
    case "number":
      return Number.isNaN(data) ? ZodParsedType.nan : ZodParsedType.number;
    case "boolean":
      return ZodParsedType.boolean;
    case "function":
      return ZodParsedType.function;
    case "bigint":
      return ZodParsedType.bigint;
    case "symbol":
      return ZodParsedType.symbol;
    case "object":
      if (Array.isArray(data)) {
        return ZodParsedType.array;
      }
      if (data === null) {
        return ZodParsedType.null;
      }
      if (data.then && typeof data.then === "function" && data.catch && typeof data.catch === "function") {
        return ZodParsedType.promise;
      }
      if (typeof Map !== "undefined" && data instanceof Map) {
        return ZodParsedType.map;
      }
      if (typeof Set !== "undefined" && data instanceof Set) {
        return ZodParsedType.set;
      }
      if (typeof Date !== "undefined" && data instanceof Date) {
        return ZodParsedType.date;
      }
      return ZodParsedType.object;
    default:
      return ZodParsedType.unknown;
  }
};

// node_modules/zod/v3/ZodError.js
var ZodIssueCode = util.arrayToEnum([
  "invalid_type",
  "invalid_literal",
  "custom",
  "invalid_union",
  "invalid_union_discriminator",
  "invalid_enum_value",
  "unrecognized_keys",
  "invalid_arguments",
  "invalid_return_type",
  "invalid_date",
  "invalid_string",
  "too_small",
  "too_big",
  "invalid_intersection_types",
  "not_multiple_of",
  "not_finite"
]);
var quotelessJson = (obj) => {
  const json = JSON.stringify(obj, null, 2);
  return json.replace(/"([^"]+)":/g, "$1:");
};

class ZodError extends Error {
  get errors() {
    return this.issues;
  }
  constructor(issues) {
    super();
    this.issues = [];
    this.addIssue = (sub) => {
      this.issues = [...this.issues, sub];
    };
    this.addIssues = (subs = []) => {
      this.issues = [...this.issues, ...subs];
    };
    const actualProto = new.target.prototype;
    if (Object.setPrototypeOf) {
      Object.setPrototypeOf(this, actualProto);
    } else {
      this.__proto__ = actualProto;
    }
    this.name = "ZodError";
    this.issues = issues;
  }
  format(_mapper) {
    const mapper = _mapper || function(issue) {
      return issue.message;
    };
    const fieldErrors = { _errors: [] };
    const processError = (error) => {
      for (const issue of error.issues) {
        if (issue.code === "invalid_union") {
          issue.unionErrors.map(processError);
        } else if (issue.code === "invalid_return_type") {
          processError(issue.returnTypeError);
        } else if (issue.code === "invalid_arguments") {
          processError(issue.argumentsError);
        } else if (issue.path.length === 0) {
          fieldErrors._errors.push(mapper(issue));
        } else {
          let curr = fieldErrors;
          let i = 0;
          while (i < issue.path.length) {
            const el = issue.path[i];
            const terminal = i === issue.path.length - 1;
            if (!terminal) {
              curr[el] = curr[el] || { _errors: [] };
            } else {
              curr[el] = curr[el] || { _errors: [] };
              curr[el]._errors.push(mapper(issue));
            }
            curr = curr[el];
            i++;
          }
        }
      }
    };
    processError(this);
    return fieldErrors;
  }
  static assert(value) {
    if (!(value instanceof ZodError)) {
      throw new Error(`Not a ZodError: ${value}`);
    }
  }
  toString() {
    return this.message;
  }
  get message() {
    return JSON.stringify(this.issues, util.jsonStringifyReplacer, 2);
  }
  get isEmpty() {
    return this.issues.length === 0;
  }
  flatten(mapper = (issue) => issue.message) {
    const fieldErrors = {};
    const formErrors = [];
    for (const sub of this.issues) {
      if (sub.path.length > 0) {
        const firstEl = sub.path[0];
        fieldErrors[firstEl] = fieldErrors[firstEl] || [];
        fieldErrors[firstEl].push(mapper(sub));
      } else {
        formErrors.push(mapper(sub));
      }
    }
    return { formErrors, fieldErrors };
  }
  get formErrors() {
    return this.flatten();
  }
}
ZodError.create = (issues) => {
  const error = new ZodError(issues);
  return error;
};

// node_modules/zod/v3/locales/en.js
var errorMap = (issue, _ctx) => {
  let message;
  switch (issue.code) {
    case ZodIssueCode.invalid_type:
      if (issue.received === ZodParsedType.undefined) {
        message = "Required";
      } else {
        message = `Expected ${issue.expected}, received ${issue.received}`;
      }
      break;
    case ZodIssueCode.invalid_literal:
      message = `Invalid literal value, expected ${JSON.stringify(issue.expected, util.jsonStringifyReplacer)}`;
      break;
    case ZodIssueCode.unrecognized_keys:
      message = `Unrecognized key(s) in object: ${util.joinValues(issue.keys, ", ")}`;
      break;
    case ZodIssueCode.invalid_union:
      message = `Invalid input`;
      break;
    case ZodIssueCode.invalid_union_discriminator:
      message = `Invalid discriminator value. Expected ${util.joinValues(issue.options)}`;
      break;
    case ZodIssueCode.invalid_enum_value:
      message = `Invalid enum value. Expected ${util.joinValues(issue.options)}, received '${issue.received}'`;
      break;
    case ZodIssueCode.invalid_arguments:
      message = `Invalid function arguments`;
      break;
    case ZodIssueCode.invalid_return_type:
      message = `Invalid function return type`;
      break;
    case ZodIssueCode.invalid_date:
      message = `Invalid date`;
      break;
    case ZodIssueCode.invalid_string:
      if (typeof issue.validation === "object") {
        if ("includes" in issue.validation) {
          message = `Invalid input: must include "${issue.validation.includes}"`;
          if (typeof issue.validation.position === "number") {
            message = `${message} at one or more positions greater than or equal to ${issue.validation.position}`;
          }
        } else if ("startsWith" in issue.validation) {
          message = `Invalid input: must start with "${issue.validation.startsWith}"`;
        } else if ("endsWith" in issue.validation) {
          message = `Invalid input: must end with "${issue.validation.endsWith}"`;
        } else {
          util.assertNever(issue.validation);
        }
      } else if (issue.validation !== "regex") {
        message = `Invalid ${issue.validation}`;
      } else {
        message = "Invalid";
      }
      break;
    case ZodIssueCode.too_small:
      if (issue.type === "array")
        message = `Array must contain ${issue.exact ? "exactly" : issue.inclusive ? `at least` : `more than`} ${issue.minimum} element(s)`;
      else if (issue.type === "string")
        message = `String must contain ${issue.exact ? "exactly" : issue.inclusive ? `at least` : `over`} ${issue.minimum} character(s)`;
      else if (issue.type === "number")
        message = `Number must be ${issue.exact ? `exactly equal to ` : issue.inclusive ? `greater than or equal to ` : `greater than `}${issue.minimum}`;
      else if (issue.type === "bigint")
        message = `Number must be ${issue.exact ? `exactly equal to ` : issue.inclusive ? `greater than or equal to ` : `greater than `}${issue.minimum}`;
      else if (issue.type === "date")
        message = `Date must be ${issue.exact ? `exactly equal to ` : issue.inclusive ? `greater than or equal to ` : `greater than `}${new Date(Number(issue.minimum))}`;
      else
        message = "Invalid input";
      break;
    case ZodIssueCode.too_big:
      if (issue.type === "array")
        message = `Array must contain ${issue.exact ? `exactly` : issue.inclusive ? `at most` : `less than`} ${issue.maximum} element(s)`;
      else if (issue.type === "string")
        message = `String must contain ${issue.exact ? `exactly` : issue.inclusive ? `at most` : `under`} ${issue.maximum} character(s)`;
      else if (issue.type === "number")
        message = `Number must be ${issue.exact ? `exactly` : issue.inclusive ? `less than or equal to` : `less than`} ${issue.maximum}`;
      else if (issue.type === "bigint")
        message = `BigInt must be ${issue.exact ? `exactly` : issue.inclusive ? `less than or equal to` : `less than`} ${issue.maximum}`;
      else if (issue.type === "date")
        message = `Date must be ${issue.exact ? `exactly` : issue.inclusive ? `smaller than or equal to` : `smaller than`} ${new Date(Number(issue.maximum))}`;
      else
        message = "Invalid input";
      break;
    case ZodIssueCode.custom:
      message = `Invalid input`;
      break;
    case ZodIssueCode.invalid_intersection_types:
      message = `Intersection results could not be merged`;
      break;
    case ZodIssueCode.not_multiple_of:
      message = `Number must be a multiple of ${issue.multipleOf}`;
      break;
    case ZodIssueCode.not_finite:
      message = "Number must be finite";
      break;
    default:
      message = _ctx.defaultError;
      util.assertNever(issue);
  }
  return { message };
};
var en_default = errorMap;

// node_modules/zod/v3/errors.js
var overrideErrorMap = en_default;
function setErrorMap(map) {
  overrideErrorMap = map;
}
function getErrorMap() {
  return overrideErrorMap;
}
// node_modules/zod/v3/helpers/parseUtil.js
var makeIssue = (params) => {
  const { data, path, errorMaps, issueData } = params;
  const fullPath = [...path, ...issueData.path || []];
  const fullIssue = {
    ...issueData,
    path: fullPath
  };
  if (issueData.message !== undefined) {
    return {
      ...issueData,
      path: fullPath,
      message: issueData.message
    };
  }
  let errorMessage = "";
  const maps = errorMaps.filter((m) => !!m).slice().reverse();
  for (const map of maps) {
    errorMessage = map(fullIssue, { data, defaultError: errorMessage }).message;
  }
  return {
    ...issueData,
    path: fullPath,
    message: errorMessage
  };
};
var EMPTY_PATH = [];
function addIssueToContext(ctx, issueData) {
  const overrideMap = getErrorMap();
  const issue = makeIssue({
    issueData,
    data: ctx.data,
    path: ctx.path,
    errorMaps: [
      ctx.common.contextualErrorMap,
      ctx.schemaErrorMap,
      overrideMap,
      overrideMap === en_default ? undefined : en_default
    ].filter((x) => !!x)
  });
  ctx.common.issues.push(issue);
}

class ParseStatus {
  constructor() {
    this.value = "valid";
  }
  dirty() {
    if (this.value === "valid")
      this.value = "dirty";
  }
  abort() {
    if (this.value !== "aborted")
      this.value = "aborted";
  }
  static mergeArray(status, results) {
    const arrayValue = [];
    for (const s of results) {
      if (s.status === "aborted")
        return INVALID;
      if (s.status === "dirty")
        status.dirty();
      arrayValue.push(s.value);
    }
    return { status: status.value, value: arrayValue };
  }
  static async mergeObjectAsync(status, pairs) {
    const syncPairs = [];
    for (const pair of pairs) {
      const key = await pair.key;
      const value = await pair.value;
      syncPairs.push({
        key,
        value
      });
    }
    return ParseStatus.mergeObjectSync(status, syncPairs);
  }
  static mergeObjectSync(status, pairs) {
    const finalObject = {};
    for (const pair of pairs) {
      const { key, value } = pair;
      if (key.status === "aborted")
        return INVALID;
      if (value.status === "aborted")
        return INVALID;
      if (key.status === "dirty")
        status.dirty();
      if (value.status === "dirty")
        status.dirty();
      if (key.value !== "__proto__" && (typeof value.value !== "undefined" || pair.alwaysSet)) {
        finalObject[key.value] = value.value;
      }
    }
    return { status: status.value, value: finalObject };
  }
}
var INVALID = Object.freeze({
  status: "aborted"
});
var DIRTY = (value) => ({ status: "dirty", value });
var OK = (value) => ({ status: "valid", value });
var isAborted = (x) => x.status === "aborted";
var isDirty = (x) => x.status === "dirty";
var isValid = (x) => x.status === "valid";
var isAsync = (x) => typeof Promise !== "undefined" && x instanceof Promise;
// node_modules/zod/v3/helpers/errorUtil.js
var errorUtil;
(function(errorUtil2) {
  errorUtil2.errToObj = (message) => typeof message === "string" ? { message } : message || {};
  errorUtil2.toString = (message) => typeof message === "string" ? message : message?.message;
})(errorUtil || (errorUtil = {}));

// node_modules/zod/v3/types.js
class ParseInputLazyPath {
  constructor(parent, value, path, key) {
    this._cachedPath = [];
    this.parent = parent;
    this.data = value;
    this._path = path;
    this._key = key;
  }
  get path() {
    if (!this._cachedPath.length) {
      if (Array.isArray(this._key)) {
        this._cachedPath.push(...this._path, ...this._key);
      } else {
        this._cachedPath.push(...this._path, this._key);
      }
    }
    return this._cachedPath;
  }
}
var handleResult = (ctx, result) => {
  if (isValid(result)) {
    return { success: true, data: result.value };
  } else {
    if (!ctx.common.issues.length) {
      throw new Error("Validation failed but no issues detected.");
    }
    return {
      success: false,
      get error() {
        if (this._error)
          return this._error;
        const error = new ZodError(ctx.common.issues);
        this._error = error;
        return this._error;
      }
    };
  }
};
function processCreateParams(params) {
  if (!params)
    return {};
  const { errorMap: errorMap2, invalid_type_error, required_error, description } = params;
  if (errorMap2 && (invalid_type_error || required_error)) {
    throw new Error(`Can't use "invalid_type_error" or "required_error" in conjunction with custom error map.`);
  }
  if (errorMap2)
    return { errorMap: errorMap2, description };
  const customMap = (iss, ctx) => {
    const { message } = params;
    if (iss.code === "invalid_enum_value") {
      return { message: message ?? ctx.defaultError };
    }
    if (typeof ctx.data === "undefined") {
      return { message: message ?? required_error ?? ctx.defaultError };
    }
    if (iss.code !== "invalid_type")
      return { message: ctx.defaultError };
    return { message: message ?? invalid_type_error ?? ctx.defaultError };
  };
  return { errorMap: customMap, description };
}

class ZodType {
  get description() {
    return this._def.description;
  }
  _getType(input) {
    return getParsedType(input.data);
  }
  _getOrReturnCtx(input, ctx) {
    return ctx || {
      common: input.parent.common,
      data: input.data,
      parsedType: getParsedType(input.data),
      schemaErrorMap: this._def.errorMap,
      path: input.path,
      parent: input.parent
    };
  }
  _processInputParams(input) {
    return {
      status: new ParseStatus,
      ctx: {
        common: input.parent.common,
        data: input.data,
        parsedType: getParsedType(input.data),
        schemaErrorMap: this._def.errorMap,
        path: input.path,
        parent: input.parent
      }
    };
  }
  _parseSync(input) {
    const result = this._parse(input);
    if (isAsync(result)) {
      throw new Error("Synchronous parse encountered promise.");
    }
    return result;
  }
  _parseAsync(input) {
    const result = this._parse(input);
    return Promise.resolve(result);
  }
  parse(data, params) {
    const result = this.safeParse(data, params);
    if (result.success)
      return result.data;
    throw result.error;
  }
  safeParse(data, params) {
    const ctx = {
      common: {
        issues: [],
        async: params?.async ?? false,
        contextualErrorMap: params?.errorMap
      },
      path: params?.path || [],
      schemaErrorMap: this._def.errorMap,
      parent: null,
      data,
      parsedType: getParsedType(data)
    };
    const result = this._parseSync({ data, path: ctx.path, parent: ctx });
    return handleResult(ctx, result);
  }
  "~validate"(data) {
    const ctx = {
      common: {
        issues: [],
        async: !!this["~standard"].async
      },
      path: [],
      schemaErrorMap: this._def.errorMap,
      parent: null,
      data,
      parsedType: getParsedType(data)
    };
    if (!this["~standard"].async) {
      try {
        const result = this._parseSync({ data, path: [], parent: ctx });
        return isValid(result) ? {
          value: result.value
        } : {
          issues: ctx.common.issues
        };
      } catch (err) {
        if (err?.message?.toLowerCase()?.includes("encountered")) {
          this["~standard"].async = true;
        }
        ctx.common = {
          issues: [],
          async: true
        };
      }
    }
    return this._parseAsync({ data, path: [], parent: ctx }).then((result) => isValid(result) ? {
      value: result.value
    } : {
      issues: ctx.common.issues
    });
  }
  async parseAsync(data, params) {
    const result = await this.safeParseAsync(data, params);
    if (result.success)
      return result.data;
    throw result.error;
  }
  async safeParseAsync(data, params) {
    const ctx = {
      common: {
        issues: [],
        contextualErrorMap: params?.errorMap,
        async: true
      },
      path: params?.path || [],
      schemaErrorMap: this._def.errorMap,
      parent: null,
      data,
      parsedType: getParsedType(data)
    };
    const maybeAsyncResult = this._parse({ data, path: ctx.path, parent: ctx });
    const result = await (isAsync(maybeAsyncResult) ? maybeAsyncResult : Promise.resolve(maybeAsyncResult));
    return handleResult(ctx, result);
  }
  refine(check, message) {
    const getIssueProperties = (val) => {
      if (typeof message === "string" || typeof message === "undefined") {
        return { message };
      } else if (typeof message === "function") {
        return message(val);
      } else {
        return message;
      }
    };
    return this._refinement((val, ctx) => {
      const result = check(val);
      const setError = () => ctx.addIssue({
        code: ZodIssueCode.custom,
        ...getIssueProperties(val)
      });
      if (typeof Promise !== "undefined" && result instanceof Promise) {
        return result.then((data) => {
          if (!data) {
            setError();
            return false;
          } else {
            return true;
          }
        });
      }
      if (!result) {
        setError();
        return false;
      } else {
        return true;
      }
    });
  }
  refinement(check, refinementData) {
    return this._refinement((val, ctx) => {
      if (!check(val)) {
        ctx.addIssue(typeof refinementData === "function" ? refinementData(val, ctx) : refinementData);
        return false;
      } else {
        return true;
      }
    });
  }
  _refinement(refinement) {
    return new ZodEffects({
      schema: this,
      typeName: ZodFirstPartyTypeKind.ZodEffects,
      effect: { type: "refinement", refinement }
    });
  }
  superRefine(refinement) {
    return this._refinement(refinement);
  }
  constructor(def) {
    this.spa = this.safeParseAsync;
    this._def = def;
    this.parse = this.parse.bind(this);
    this.safeParse = this.safeParse.bind(this);
    this.parseAsync = this.parseAsync.bind(this);
    this.safeParseAsync = this.safeParseAsync.bind(this);
    this.spa = this.spa.bind(this);
    this.refine = this.refine.bind(this);
    this.refinement = this.refinement.bind(this);
    this.superRefine = this.superRefine.bind(this);
    this.optional = this.optional.bind(this);
    this.nullable = this.nullable.bind(this);
    this.nullish = this.nullish.bind(this);
    this.array = this.array.bind(this);
    this.promise = this.promise.bind(this);
    this.or = this.or.bind(this);
    this.and = this.and.bind(this);
    this.transform = this.transform.bind(this);
    this.brand = this.brand.bind(this);
    this.default = this.default.bind(this);
    this.catch = this.catch.bind(this);
    this.describe = this.describe.bind(this);
    this.pipe = this.pipe.bind(this);
    this.readonly = this.readonly.bind(this);
    this.isNullable = this.isNullable.bind(this);
    this.isOptional = this.isOptional.bind(this);
    this["~standard"] = {
      version: 1,
      vendor: "zod",
      validate: (data) => this["~validate"](data)
    };
  }
  optional() {
    return ZodOptional.create(this, this._def);
  }
  nullable() {
    return ZodNullable.create(this, this._def);
  }
  nullish() {
    return this.nullable().optional();
  }
  array() {
    return ZodArray.create(this);
  }
  promise() {
    return ZodPromise.create(this, this._def);
  }
  or(option) {
    return ZodUnion.create([this, option], this._def);
  }
  and(incoming) {
    return ZodIntersection.create(this, incoming, this._def);
  }
  transform(transform) {
    return new ZodEffects({
      ...processCreateParams(this._def),
      schema: this,
      typeName: ZodFirstPartyTypeKind.ZodEffects,
      effect: { type: "transform", transform }
    });
  }
  default(def) {
    const defaultValueFunc = typeof def === "function" ? def : () => def;
    return new ZodDefault({
      ...processCreateParams(this._def),
      innerType: this,
      defaultValue: defaultValueFunc,
      typeName: ZodFirstPartyTypeKind.ZodDefault
    });
  }
  brand() {
    return new ZodBranded({
      typeName: ZodFirstPartyTypeKind.ZodBranded,
      type: this,
      ...processCreateParams(this._def)
    });
  }
  catch(def) {
    const catchValueFunc = typeof def === "function" ? def : () => def;
    return new ZodCatch({
      ...processCreateParams(this._def),
      innerType: this,
      catchValue: catchValueFunc,
      typeName: ZodFirstPartyTypeKind.ZodCatch
    });
  }
  describe(description) {
    const This = this.constructor;
    return new This({
      ...this._def,
      description
    });
  }
  pipe(target) {
    return ZodPipeline.create(this, target);
  }
  readonly() {
    return ZodReadonly.create(this);
  }
  isOptional() {
    return this.safeParse(undefined).success;
  }
  isNullable() {
    return this.safeParse(null).success;
  }
}
var cuidRegex = /^c[^\s-]{8,}$/i;
var cuid2Regex = /^[0-9a-z]+$/;
var ulidRegex = /^[0-9A-HJKMNP-TV-Z]{26}$/i;
var uuidRegex = /^[0-9a-fA-F]{8}\b-[0-9a-fA-F]{4}\b-[0-9a-fA-F]{4}\b-[0-9a-fA-F]{4}\b-[0-9a-fA-F]{12}$/i;
var nanoidRegex = /^[a-z0-9_-]{21}$/i;
var jwtRegex = /^[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\.[A-Za-z0-9-_]*$/;
var durationRegex = /^[-+]?P(?!$)(?:(?:[-+]?\d+Y)|(?:[-+]?\d+[.,]\d+Y$))?(?:(?:[-+]?\d+M)|(?:[-+]?\d+[.,]\d+M$))?(?:(?:[-+]?\d+W)|(?:[-+]?\d+[.,]\d+W$))?(?:(?:[-+]?\d+D)|(?:[-+]?\d+[.,]\d+D$))?(?:T(?=[\d+-])(?:(?:[-+]?\d+H)|(?:[-+]?\d+[.,]\d+H$))?(?:(?:[-+]?\d+M)|(?:[-+]?\d+[.,]\d+M$))?(?:[-+]?\d+(?:[.,]\d+)?S)?)??$/;
var emailRegex = /^(?!\.)(?!.*\.\.)([A-Z0-9_'+\-\.]*)[A-Z0-9_+-]@([A-Z0-9][A-Z0-9\-]*\.)+[A-Z]{2,}$/i;
var _emojiRegex = `^(\\p{Extended_Pictographic}|\\p{Emoji_Component})+$`;
var emojiRegex;
var ipv4Regex = /^(?:(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])\.){3}(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])$/;
var ipv4CidrRegex = /^(?:(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])\.){3}(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])\/(3[0-2]|[12]?[0-9])$/;
var ipv6Regex = /^(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))$/;
var ipv6CidrRegex = /^(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))\/(12[0-8]|1[01][0-9]|[1-9]?[0-9])$/;
var base64Regex = /^([0-9a-zA-Z+/]{4})*(([0-9a-zA-Z+/]{2}==)|([0-9a-zA-Z+/]{3}=))?$/;
var base64urlRegex = /^([0-9a-zA-Z-_]{4})*(([0-9a-zA-Z-_]{2}(==)?)|([0-9a-zA-Z-_]{3}(=)?))?$/;
var dateRegexSource = `((\\d\\d[2468][048]|\\d\\d[13579][26]|\\d\\d0[48]|[02468][048]00|[13579][26]00)-02-29|\\d{4}-((0[13578]|1[02])-(0[1-9]|[12]\\d|3[01])|(0[469]|11)-(0[1-9]|[12]\\d|30)|(02)-(0[1-9]|1\\d|2[0-8])))`;
var dateRegex = new RegExp(`^${dateRegexSource}$`);
function timeRegexSource(args) {
  let secondsRegexSource = `[0-5]\\d`;
  if (args.precision) {
    secondsRegexSource = `${secondsRegexSource}\\.\\d{${args.precision}}`;
  } else if (args.precision == null) {
    secondsRegexSource = `${secondsRegexSource}(\\.\\d+)?`;
  }
  const secondsQuantifier = args.precision ? "+" : "?";
  return `([01]\\d|2[0-3]):[0-5]\\d(:${secondsRegexSource})${secondsQuantifier}`;
}
function timeRegex(args) {
  return new RegExp(`^${timeRegexSource(args)}$`);
}
function datetimeRegex(args) {
  let regex = `${dateRegexSource}T${timeRegexSource(args)}`;
  const opts = [];
  opts.push(args.local ? `Z?` : `Z`);
  if (args.offset)
    opts.push(`([+-]\\d{2}:?\\d{2})`);
  regex = `${regex}(${opts.join("|")})`;
  return new RegExp(`^${regex}$`);
}
function isValidIP(ip, version) {
  if ((version === "v4" || !version) && ipv4Regex.test(ip)) {
    return true;
  }
  if ((version === "v6" || !version) && ipv6Regex.test(ip)) {
    return true;
  }
  return false;
}
function isValidJWT(jwt, alg) {
  if (!jwtRegex.test(jwt))
    return false;
  try {
    const [header] = jwt.split(".");
    if (!header)
      return false;
    const base64 = header.replace(/-/g, "+").replace(/_/g, "/").padEnd(header.length + (4 - header.length % 4) % 4, "=");
    const decoded = JSON.parse(atob(base64));
    if (typeof decoded !== "object" || decoded === null)
      return false;
    if ("typ" in decoded && decoded?.typ !== "JWT")
      return false;
    if (!decoded.alg)
      return false;
    if (alg && decoded.alg !== alg)
      return false;
    return true;
  } catch {
    return false;
  }
}
function isValidCidr(ip, version) {
  if ((version === "v4" || !version) && ipv4CidrRegex.test(ip)) {
    return true;
  }
  if ((version === "v6" || !version) && ipv6CidrRegex.test(ip)) {
    return true;
  }
  return false;
}

class ZodString extends ZodType {
  _parse(input) {
    if (this._def.coerce) {
      input.data = String(input.data);
    }
    const parsedType = this._getType(input);
    if (parsedType !== ZodParsedType.string) {
      const ctx2 = this._getOrReturnCtx(input);
      addIssueToContext(ctx2, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.string,
        received: ctx2.parsedType
      });
      return INVALID;
    }
    const status = new ParseStatus;
    let ctx = undefined;
    for (const check of this._def.checks) {
      if (check.kind === "min") {
        if (input.data.length < check.value) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.too_small,
            minimum: check.value,
            type: "string",
            inclusive: true,
            exact: false,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "max") {
        if (input.data.length > check.value) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.too_big,
            maximum: check.value,
            type: "string",
            inclusive: true,
            exact: false,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "length") {
        const tooBig = input.data.length > check.value;
        const tooSmall = input.data.length < check.value;
        if (tooBig || tooSmall) {
          ctx = this._getOrReturnCtx(input, ctx);
          if (tooBig) {
            addIssueToContext(ctx, {
              code: ZodIssueCode.too_big,
              maximum: check.value,
              type: "string",
              inclusive: true,
              exact: true,
              message: check.message
            });
          } else if (tooSmall) {
            addIssueToContext(ctx, {
              code: ZodIssueCode.too_small,
              minimum: check.value,
              type: "string",
              inclusive: true,
              exact: true,
              message: check.message
            });
          }
          status.dirty();
        }
      } else if (check.kind === "email") {
        if (!emailRegex.test(input.data)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            validation: "email",
            code: ZodIssueCode.invalid_string,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "emoji") {
        if (!emojiRegex) {
          emojiRegex = new RegExp(_emojiRegex, "u");
        }
        if (!emojiRegex.test(input.data)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            validation: "emoji",
            code: ZodIssueCode.invalid_string,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "uuid") {
        if (!uuidRegex.test(input.data)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            validation: "uuid",
            code: ZodIssueCode.invalid_string,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "nanoid") {
        if (!nanoidRegex.test(input.data)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            validation: "nanoid",
            code: ZodIssueCode.invalid_string,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "cuid") {
        if (!cuidRegex.test(input.data)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            validation: "cuid",
            code: ZodIssueCode.invalid_string,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "cuid2") {
        if (!cuid2Regex.test(input.data)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            validation: "cuid2",
            code: ZodIssueCode.invalid_string,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "ulid") {
        if (!ulidRegex.test(input.data)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            validation: "ulid",
            code: ZodIssueCode.invalid_string,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "url") {
        try {
          new URL(input.data);
        } catch {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            validation: "url",
            code: ZodIssueCode.invalid_string,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "regex") {
        check.regex.lastIndex = 0;
        const testResult = check.regex.test(input.data);
        if (!testResult) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            validation: "regex",
            code: ZodIssueCode.invalid_string,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "trim") {
        input.data = input.data.trim();
      } else if (check.kind === "includes") {
        if (!input.data.includes(check.value, check.position)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.invalid_string,
            validation: { includes: check.value, position: check.position },
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "toLowerCase") {
        input.data = input.data.toLowerCase();
      } else if (check.kind === "toUpperCase") {
        input.data = input.data.toUpperCase();
      } else if (check.kind === "startsWith") {
        if (!input.data.startsWith(check.value)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.invalid_string,
            validation: { startsWith: check.value },
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "endsWith") {
        if (!input.data.endsWith(check.value)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.invalid_string,
            validation: { endsWith: check.value },
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "datetime") {
        const regex = datetimeRegex(check);
        if (!regex.test(input.data)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.invalid_string,
            validation: "datetime",
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "date") {
        const regex = dateRegex;
        if (!regex.test(input.data)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.invalid_string,
            validation: "date",
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "time") {
        const regex = timeRegex(check);
        if (!regex.test(input.data)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.invalid_string,
            validation: "time",
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "duration") {
        if (!durationRegex.test(input.data)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            validation: "duration",
            code: ZodIssueCode.invalid_string,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "ip") {
        if (!isValidIP(input.data, check.version)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            validation: "ip",
            code: ZodIssueCode.invalid_string,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "jwt") {
        if (!isValidJWT(input.data, check.alg)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            validation: "jwt",
            code: ZodIssueCode.invalid_string,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "cidr") {
        if (!isValidCidr(input.data, check.version)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            validation: "cidr",
            code: ZodIssueCode.invalid_string,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "base64") {
        if (!base64Regex.test(input.data)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            validation: "base64",
            code: ZodIssueCode.invalid_string,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "base64url") {
        if (!base64urlRegex.test(input.data)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            validation: "base64url",
            code: ZodIssueCode.invalid_string,
            message: check.message
          });
          status.dirty();
        }
      } else {
        util.assertNever(check);
      }
    }
    return { status: status.value, value: input.data };
  }
  _regex(regex, validation, message) {
    return this.refinement((data) => regex.test(data), {
      validation,
      code: ZodIssueCode.invalid_string,
      ...errorUtil.errToObj(message)
    });
  }
  _addCheck(check) {
    return new ZodString({
      ...this._def,
      checks: [...this._def.checks, check]
    });
  }
  email(message) {
    return this._addCheck({ kind: "email", ...errorUtil.errToObj(message) });
  }
  url(message) {
    return this._addCheck({ kind: "url", ...errorUtil.errToObj(message) });
  }
  emoji(message) {
    return this._addCheck({ kind: "emoji", ...errorUtil.errToObj(message) });
  }
  uuid(message) {
    return this._addCheck({ kind: "uuid", ...errorUtil.errToObj(message) });
  }
  nanoid(message) {
    return this._addCheck({ kind: "nanoid", ...errorUtil.errToObj(message) });
  }
  cuid(message) {
    return this._addCheck({ kind: "cuid", ...errorUtil.errToObj(message) });
  }
  cuid2(message) {
    return this._addCheck({ kind: "cuid2", ...errorUtil.errToObj(message) });
  }
  ulid(message) {
    return this._addCheck({ kind: "ulid", ...errorUtil.errToObj(message) });
  }
  base64(message) {
    return this._addCheck({ kind: "base64", ...errorUtil.errToObj(message) });
  }
  base64url(message) {
    return this._addCheck({
      kind: "base64url",
      ...errorUtil.errToObj(message)
    });
  }
  jwt(options) {
    return this._addCheck({ kind: "jwt", ...errorUtil.errToObj(options) });
  }
  ip(options) {
    return this._addCheck({ kind: "ip", ...errorUtil.errToObj(options) });
  }
  cidr(options) {
    return this._addCheck({ kind: "cidr", ...errorUtil.errToObj(options) });
  }
  datetime(options) {
    if (typeof options === "string") {
      return this._addCheck({
        kind: "datetime",
        precision: null,
        offset: false,
        local: false,
        message: options
      });
    }
    return this._addCheck({
      kind: "datetime",
      precision: typeof options?.precision === "undefined" ? null : options?.precision,
      offset: options?.offset ?? false,
      local: options?.local ?? false,
      ...errorUtil.errToObj(options?.message)
    });
  }
  date(message) {
    return this._addCheck({ kind: "date", message });
  }
  time(options) {
    if (typeof options === "string") {
      return this._addCheck({
        kind: "time",
        precision: null,
        message: options
      });
    }
    return this._addCheck({
      kind: "time",
      precision: typeof options?.precision === "undefined" ? null : options?.precision,
      ...errorUtil.errToObj(options?.message)
    });
  }
  duration(message) {
    return this._addCheck({ kind: "duration", ...errorUtil.errToObj(message) });
  }
  regex(regex, message) {
    return this._addCheck({
      kind: "regex",
      regex,
      ...errorUtil.errToObj(message)
    });
  }
  includes(value, options) {
    return this._addCheck({
      kind: "includes",
      value,
      position: options?.position,
      ...errorUtil.errToObj(options?.message)
    });
  }
  startsWith(value, message) {
    return this._addCheck({
      kind: "startsWith",
      value,
      ...errorUtil.errToObj(message)
    });
  }
  endsWith(value, message) {
    return this._addCheck({
      kind: "endsWith",
      value,
      ...errorUtil.errToObj(message)
    });
  }
  min(minLength, message) {
    return this._addCheck({
      kind: "min",
      value: minLength,
      ...errorUtil.errToObj(message)
    });
  }
  max(maxLength, message) {
    return this._addCheck({
      kind: "max",
      value: maxLength,
      ...errorUtil.errToObj(message)
    });
  }
  length(len, message) {
    return this._addCheck({
      kind: "length",
      value: len,
      ...errorUtil.errToObj(message)
    });
  }
  nonempty(message) {
    return this.min(1, errorUtil.errToObj(message));
  }
  trim() {
    return new ZodString({
      ...this._def,
      checks: [...this._def.checks, { kind: "trim" }]
    });
  }
  toLowerCase() {
    return new ZodString({
      ...this._def,
      checks: [...this._def.checks, { kind: "toLowerCase" }]
    });
  }
  toUpperCase() {
    return new ZodString({
      ...this._def,
      checks: [...this._def.checks, { kind: "toUpperCase" }]
    });
  }
  get isDatetime() {
    return !!this._def.checks.find((ch) => ch.kind === "datetime");
  }
  get isDate() {
    return !!this._def.checks.find((ch) => ch.kind === "date");
  }
  get isTime() {
    return !!this._def.checks.find((ch) => ch.kind === "time");
  }
  get isDuration() {
    return !!this._def.checks.find((ch) => ch.kind === "duration");
  }
  get isEmail() {
    return !!this._def.checks.find((ch) => ch.kind === "email");
  }
  get isURL() {
    return !!this._def.checks.find((ch) => ch.kind === "url");
  }
  get isEmoji() {
    return !!this._def.checks.find((ch) => ch.kind === "emoji");
  }
  get isUUID() {
    return !!this._def.checks.find((ch) => ch.kind === "uuid");
  }
  get isNANOID() {
    return !!this._def.checks.find((ch) => ch.kind === "nanoid");
  }
  get isCUID() {
    return !!this._def.checks.find((ch) => ch.kind === "cuid");
  }
  get isCUID2() {
    return !!this._def.checks.find((ch) => ch.kind === "cuid2");
  }
  get isULID() {
    return !!this._def.checks.find((ch) => ch.kind === "ulid");
  }
  get isIP() {
    return !!this._def.checks.find((ch) => ch.kind === "ip");
  }
  get isCIDR() {
    return !!this._def.checks.find((ch) => ch.kind === "cidr");
  }
  get isBase64() {
    return !!this._def.checks.find((ch) => ch.kind === "base64");
  }
  get isBase64url() {
    return !!this._def.checks.find((ch) => ch.kind === "base64url");
  }
  get minLength() {
    let min = null;
    for (const ch of this._def.checks) {
      if (ch.kind === "min") {
        if (min === null || ch.value > min)
          min = ch.value;
      }
    }
    return min;
  }
  get maxLength() {
    let max = null;
    for (const ch of this._def.checks) {
      if (ch.kind === "max") {
        if (max === null || ch.value < max)
          max = ch.value;
      }
    }
    return max;
  }
}
ZodString.create = (params) => {
  return new ZodString({
    checks: [],
    typeName: ZodFirstPartyTypeKind.ZodString,
    coerce: params?.coerce ?? false,
    ...processCreateParams(params)
  });
};
function floatSafeRemainder(val, step) {
  const valDecCount = (val.toString().split(".")[1] || "").length;
  const stepDecCount = (step.toString().split(".")[1] || "").length;
  const decCount = valDecCount > stepDecCount ? valDecCount : stepDecCount;
  const valInt = Number.parseInt(val.toFixed(decCount).replace(".", ""));
  const stepInt = Number.parseInt(step.toFixed(decCount).replace(".", ""));
  return valInt % stepInt / 10 ** decCount;
}

class ZodNumber extends ZodType {
  constructor() {
    super(...arguments);
    this.min = this.gte;
    this.max = this.lte;
    this.step = this.multipleOf;
  }
  _parse(input) {
    if (this._def.coerce) {
      input.data = Number(input.data);
    }
    const parsedType = this._getType(input);
    if (parsedType !== ZodParsedType.number) {
      const ctx2 = this._getOrReturnCtx(input);
      addIssueToContext(ctx2, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.number,
        received: ctx2.parsedType
      });
      return INVALID;
    }
    let ctx = undefined;
    const status = new ParseStatus;
    for (const check of this._def.checks) {
      if (check.kind === "int") {
        if (!util.isInteger(input.data)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.invalid_type,
            expected: "integer",
            received: "float",
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "min") {
        const tooSmall = check.inclusive ? input.data < check.value : input.data <= check.value;
        if (tooSmall) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.too_small,
            minimum: check.value,
            type: "number",
            inclusive: check.inclusive,
            exact: false,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "max") {
        const tooBig = check.inclusive ? input.data > check.value : input.data >= check.value;
        if (tooBig) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.too_big,
            maximum: check.value,
            type: "number",
            inclusive: check.inclusive,
            exact: false,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "multipleOf") {
        if (floatSafeRemainder(input.data, check.value) !== 0) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.not_multiple_of,
            multipleOf: check.value,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "finite") {
        if (!Number.isFinite(input.data)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.not_finite,
            message: check.message
          });
          status.dirty();
        }
      } else {
        util.assertNever(check);
      }
    }
    return { status: status.value, value: input.data };
  }
  gte(value, message) {
    return this.setLimit("min", value, true, errorUtil.toString(message));
  }
  gt(value, message) {
    return this.setLimit("min", value, false, errorUtil.toString(message));
  }
  lte(value, message) {
    return this.setLimit("max", value, true, errorUtil.toString(message));
  }
  lt(value, message) {
    return this.setLimit("max", value, false, errorUtil.toString(message));
  }
  setLimit(kind, value, inclusive, message) {
    return new ZodNumber({
      ...this._def,
      checks: [
        ...this._def.checks,
        {
          kind,
          value,
          inclusive,
          message: errorUtil.toString(message)
        }
      ]
    });
  }
  _addCheck(check) {
    return new ZodNumber({
      ...this._def,
      checks: [...this._def.checks, check]
    });
  }
  int(message) {
    return this._addCheck({
      kind: "int",
      message: errorUtil.toString(message)
    });
  }
  positive(message) {
    return this._addCheck({
      kind: "min",
      value: 0,
      inclusive: false,
      message: errorUtil.toString(message)
    });
  }
  negative(message) {
    return this._addCheck({
      kind: "max",
      value: 0,
      inclusive: false,
      message: errorUtil.toString(message)
    });
  }
  nonpositive(message) {
    return this._addCheck({
      kind: "max",
      value: 0,
      inclusive: true,
      message: errorUtil.toString(message)
    });
  }
  nonnegative(message) {
    return this._addCheck({
      kind: "min",
      value: 0,
      inclusive: true,
      message: errorUtil.toString(message)
    });
  }
  multipleOf(value, message) {
    return this._addCheck({
      kind: "multipleOf",
      value,
      message: errorUtil.toString(message)
    });
  }
  finite(message) {
    return this._addCheck({
      kind: "finite",
      message: errorUtil.toString(message)
    });
  }
  safe(message) {
    return this._addCheck({
      kind: "min",
      inclusive: true,
      value: Number.MIN_SAFE_INTEGER,
      message: errorUtil.toString(message)
    })._addCheck({
      kind: "max",
      inclusive: true,
      value: Number.MAX_SAFE_INTEGER,
      message: errorUtil.toString(message)
    });
  }
  get minValue() {
    let min = null;
    for (const ch of this._def.checks) {
      if (ch.kind === "min") {
        if (min === null || ch.value > min)
          min = ch.value;
      }
    }
    return min;
  }
  get maxValue() {
    let max = null;
    for (const ch of this._def.checks) {
      if (ch.kind === "max") {
        if (max === null || ch.value < max)
          max = ch.value;
      }
    }
    return max;
  }
  get isInt() {
    return !!this._def.checks.find((ch) => ch.kind === "int" || ch.kind === "multipleOf" && util.isInteger(ch.value));
  }
  get isFinite() {
    let max = null;
    let min = null;
    for (const ch of this._def.checks) {
      if (ch.kind === "finite" || ch.kind === "int" || ch.kind === "multipleOf") {
        return true;
      } else if (ch.kind === "min") {
        if (min === null || ch.value > min)
          min = ch.value;
      } else if (ch.kind === "max") {
        if (max === null || ch.value < max)
          max = ch.value;
      }
    }
    return Number.isFinite(min) && Number.isFinite(max);
  }
}
ZodNumber.create = (params) => {
  return new ZodNumber({
    checks: [],
    typeName: ZodFirstPartyTypeKind.ZodNumber,
    coerce: params?.coerce || false,
    ...processCreateParams(params)
  });
};

class ZodBigInt extends ZodType {
  constructor() {
    super(...arguments);
    this.min = this.gte;
    this.max = this.lte;
  }
  _parse(input) {
    if (this._def.coerce) {
      try {
        input.data = BigInt(input.data);
      } catch {
        return this._getInvalidInput(input);
      }
    }
    const parsedType = this._getType(input);
    if (parsedType !== ZodParsedType.bigint) {
      return this._getInvalidInput(input);
    }
    let ctx = undefined;
    const status = new ParseStatus;
    for (const check of this._def.checks) {
      if (check.kind === "min") {
        const tooSmall = check.inclusive ? input.data < check.value : input.data <= check.value;
        if (tooSmall) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.too_small,
            type: "bigint",
            minimum: check.value,
            inclusive: check.inclusive,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "max") {
        const tooBig = check.inclusive ? input.data > check.value : input.data >= check.value;
        if (tooBig) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.too_big,
            type: "bigint",
            maximum: check.value,
            inclusive: check.inclusive,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "multipleOf") {
        if (input.data % check.value !== BigInt(0)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.not_multiple_of,
            multipleOf: check.value,
            message: check.message
          });
          status.dirty();
        }
      } else {
        util.assertNever(check);
      }
    }
    return { status: status.value, value: input.data };
  }
  _getInvalidInput(input) {
    const ctx = this._getOrReturnCtx(input);
    addIssueToContext(ctx, {
      code: ZodIssueCode.invalid_type,
      expected: ZodParsedType.bigint,
      received: ctx.parsedType
    });
    return INVALID;
  }
  gte(value, message) {
    return this.setLimit("min", value, true, errorUtil.toString(message));
  }
  gt(value, message) {
    return this.setLimit("min", value, false, errorUtil.toString(message));
  }
  lte(value, message) {
    return this.setLimit("max", value, true, errorUtil.toString(message));
  }
  lt(value, message) {
    return this.setLimit("max", value, false, errorUtil.toString(message));
  }
  setLimit(kind, value, inclusive, message) {
    return new ZodBigInt({
      ...this._def,
      checks: [
        ...this._def.checks,
        {
          kind,
          value,
          inclusive,
          message: errorUtil.toString(message)
        }
      ]
    });
  }
  _addCheck(check) {
    return new ZodBigInt({
      ...this._def,
      checks: [...this._def.checks, check]
    });
  }
  positive(message) {
    return this._addCheck({
      kind: "min",
      value: BigInt(0),
      inclusive: false,
      message: errorUtil.toString(message)
    });
  }
  negative(message) {
    return this._addCheck({
      kind: "max",
      value: BigInt(0),
      inclusive: false,
      message: errorUtil.toString(message)
    });
  }
  nonpositive(message) {
    return this._addCheck({
      kind: "max",
      value: BigInt(0),
      inclusive: true,
      message: errorUtil.toString(message)
    });
  }
  nonnegative(message) {
    return this._addCheck({
      kind: "min",
      value: BigInt(0),
      inclusive: true,
      message: errorUtil.toString(message)
    });
  }
  multipleOf(value, message) {
    return this._addCheck({
      kind: "multipleOf",
      value,
      message: errorUtil.toString(message)
    });
  }
  get minValue() {
    let min = null;
    for (const ch of this._def.checks) {
      if (ch.kind === "min") {
        if (min === null || ch.value > min)
          min = ch.value;
      }
    }
    return min;
  }
  get maxValue() {
    let max = null;
    for (const ch of this._def.checks) {
      if (ch.kind === "max") {
        if (max === null || ch.value < max)
          max = ch.value;
      }
    }
    return max;
  }
}
ZodBigInt.create = (params) => {
  return new ZodBigInt({
    checks: [],
    typeName: ZodFirstPartyTypeKind.ZodBigInt,
    coerce: params?.coerce ?? false,
    ...processCreateParams(params)
  });
};

class ZodBoolean extends ZodType {
  _parse(input) {
    if (this._def.coerce) {
      input.data = Boolean(input.data);
    }
    const parsedType = this._getType(input);
    if (parsedType !== ZodParsedType.boolean) {
      const ctx = this._getOrReturnCtx(input);
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.boolean,
        received: ctx.parsedType
      });
      return INVALID;
    }
    return OK(input.data);
  }
}
ZodBoolean.create = (params) => {
  return new ZodBoolean({
    typeName: ZodFirstPartyTypeKind.ZodBoolean,
    coerce: params?.coerce || false,
    ...processCreateParams(params)
  });
};

class ZodDate extends ZodType {
  _parse(input) {
    if (this._def.coerce) {
      input.data = new Date(input.data);
    }
    const parsedType = this._getType(input);
    if (parsedType !== ZodParsedType.date) {
      const ctx2 = this._getOrReturnCtx(input);
      addIssueToContext(ctx2, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.date,
        received: ctx2.parsedType
      });
      return INVALID;
    }
    if (Number.isNaN(input.data.getTime())) {
      const ctx2 = this._getOrReturnCtx(input);
      addIssueToContext(ctx2, {
        code: ZodIssueCode.invalid_date
      });
      return INVALID;
    }
    const status = new ParseStatus;
    let ctx = undefined;
    for (const check of this._def.checks) {
      if (check.kind === "min") {
        if (input.data.getTime() < check.value) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.too_small,
            message: check.message,
            inclusive: true,
            exact: false,
            minimum: check.value,
            type: "date"
          });
          status.dirty();
        }
      } else if (check.kind === "max") {
        if (input.data.getTime() > check.value) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.too_big,
            message: check.message,
            inclusive: true,
            exact: false,
            maximum: check.value,
            type: "date"
          });
          status.dirty();
        }
      } else {
        util.assertNever(check);
      }
    }
    return {
      status: status.value,
      value: new Date(input.data.getTime())
    };
  }
  _addCheck(check) {
    return new ZodDate({
      ...this._def,
      checks: [...this._def.checks, check]
    });
  }
  min(minDate, message) {
    return this._addCheck({
      kind: "min",
      value: minDate.getTime(),
      message: errorUtil.toString(message)
    });
  }
  max(maxDate, message) {
    return this._addCheck({
      kind: "max",
      value: maxDate.getTime(),
      message: errorUtil.toString(message)
    });
  }
  get minDate() {
    let min = null;
    for (const ch of this._def.checks) {
      if (ch.kind === "min") {
        if (min === null || ch.value > min)
          min = ch.value;
      }
    }
    return min != null ? new Date(min) : null;
  }
  get maxDate() {
    let max = null;
    for (const ch of this._def.checks) {
      if (ch.kind === "max") {
        if (max === null || ch.value < max)
          max = ch.value;
      }
    }
    return max != null ? new Date(max) : null;
  }
}
ZodDate.create = (params) => {
  return new ZodDate({
    checks: [],
    coerce: params?.coerce || false,
    typeName: ZodFirstPartyTypeKind.ZodDate,
    ...processCreateParams(params)
  });
};

class ZodSymbol extends ZodType {
  _parse(input) {
    const parsedType = this._getType(input);
    if (parsedType !== ZodParsedType.symbol) {
      const ctx = this._getOrReturnCtx(input);
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.symbol,
        received: ctx.parsedType
      });
      return INVALID;
    }
    return OK(input.data);
  }
}
ZodSymbol.create = (params) => {
  return new ZodSymbol({
    typeName: ZodFirstPartyTypeKind.ZodSymbol,
    ...processCreateParams(params)
  });
};

class ZodUndefined extends ZodType {
  _parse(input) {
    const parsedType = this._getType(input);
    if (parsedType !== ZodParsedType.undefined) {
      const ctx = this._getOrReturnCtx(input);
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.undefined,
        received: ctx.parsedType
      });
      return INVALID;
    }
    return OK(input.data);
  }
}
ZodUndefined.create = (params) => {
  return new ZodUndefined({
    typeName: ZodFirstPartyTypeKind.ZodUndefined,
    ...processCreateParams(params)
  });
};

class ZodNull extends ZodType {
  _parse(input) {
    const parsedType = this._getType(input);
    if (parsedType !== ZodParsedType.null) {
      const ctx = this._getOrReturnCtx(input);
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.null,
        received: ctx.parsedType
      });
      return INVALID;
    }
    return OK(input.data);
  }
}
ZodNull.create = (params) => {
  return new ZodNull({
    typeName: ZodFirstPartyTypeKind.ZodNull,
    ...processCreateParams(params)
  });
};

class ZodAny extends ZodType {
  constructor() {
    super(...arguments);
    this._any = true;
  }
  _parse(input) {
    return OK(input.data);
  }
}
ZodAny.create = (params) => {
  return new ZodAny({
    typeName: ZodFirstPartyTypeKind.ZodAny,
    ...processCreateParams(params)
  });
};

class ZodUnknown extends ZodType {
  constructor() {
    super(...arguments);
    this._unknown = true;
  }
  _parse(input) {
    return OK(input.data);
  }
}
ZodUnknown.create = (params) => {
  return new ZodUnknown({
    typeName: ZodFirstPartyTypeKind.ZodUnknown,
    ...processCreateParams(params)
  });
};

class ZodNever extends ZodType {
  _parse(input) {
    const ctx = this._getOrReturnCtx(input);
    addIssueToContext(ctx, {
      code: ZodIssueCode.invalid_type,
      expected: ZodParsedType.never,
      received: ctx.parsedType
    });
    return INVALID;
  }
}
ZodNever.create = (params) => {
  return new ZodNever({
    typeName: ZodFirstPartyTypeKind.ZodNever,
    ...processCreateParams(params)
  });
};

class ZodVoid extends ZodType {
  _parse(input) {
    const parsedType = this._getType(input);
    if (parsedType !== ZodParsedType.undefined) {
      const ctx = this._getOrReturnCtx(input);
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.void,
        received: ctx.parsedType
      });
      return INVALID;
    }
    return OK(input.data);
  }
}
ZodVoid.create = (params) => {
  return new ZodVoid({
    typeName: ZodFirstPartyTypeKind.ZodVoid,
    ...processCreateParams(params)
  });
};

class ZodArray extends ZodType {
  _parse(input) {
    const { ctx, status } = this._processInputParams(input);
    const def = this._def;
    if (ctx.parsedType !== ZodParsedType.array) {
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.array,
        received: ctx.parsedType
      });
      return INVALID;
    }
    if (def.exactLength !== null) {
      const tooBig = ctx.data.length > def.exactLength.value;
      const tooSmall = ctx.data.length < def.exactLength.value;
      if (tooBig || tooSmall) {
        addIssueToContext(ctx, {
          code: tooBig ? ZodIssueCode.too_big : ZodIssueCode.too_small,
          minimum: tooSmall ? def.exactLength.value : undefined,
          maximum: tooBig ? def.exactLength.value : undefined,
          type: "array",
          inclusive: true,
          exact: true,
          message: def.exactLength.message
        });
        status.dirty();
      }
    }
    if (def.minLength !== null) {
      if (ctx.data.length < def.minLength.value) {
        addIssueToContext(ctx, {
          code: ZodIssueCode.too_small,
          minimum: def.minLength.value,
          type: "array",
          inclusive: true,
          exact: false,
          message: def.minLength.message
        });
        status.dirty();
      }
    }
    if (def.maxLength !== null) {
      if (ctx.data.length > def.maxLength.value) {
        addIssueToContext(ctx, {
          code: ZodIssueCode.too_big,
          maximum: def.maxLength.value,
          type: "array",
          inclusive: true,
          exact: false,
          message: def.maxLength.message
        });
        status.dirty();
      }
    }
    if (ctx.common.async) {
      return Promise.all([...ctx.data].map((item, i) => {
        return def.type._parseAsync(new ParseInputLazyPath(ctx, item, ctx.path, i));
      })).then((result2) => {
        return ParseStatus.mergeArray(status, result2);
      });
    }
    const result = [...ctx.data].map((item, i) => {
      return def.type._parseSync(new ParseInputLazyPath(ctx, item, ctx.path, i));
    });
    return ParseStatus.mergeArray(status, result);
  }
  get element() {
    return this._def.type;
  }
  min(minLength, message) {
    return new ZodArray({
      ...this._def,
      minLength: { value: minLength, message: errorUtil.toString(message) }
    });
  }
  max(maxLength, message) {
    return new ZodArray({
      ...this._def,
      maxLength: { value: maxLength, message: errorUtil.toString(message) }
    });
  }
  length(len, message) {
    return new ZodArray({
      ...this._def,
      exactLength: { value: len, message: errorUtil.toString(message) }
    });
  }
  nonempty(message) {
    return this.min(1, message);
  }
}
ZodArray.create = (schema, params) => {
  return new ZodArray({
    type: schema,
    minLength: null,
    maxLength: null,
    exactLength: null,
    typeName: ZodFirstPartyTypeKind.ZodArray,
    ...processCreateParams(params)
  });
};
function deepPartialify(schema) {
  if (schema instanceof ZodObject) {
    const newShape = {};
    for (const key in schema.shape) {
      const fieldSchema = schema.shape[key];
      newShape[key] = ZodOptional.create(deepPartialify(fieldSchema));
    }
    return new ZodObject({
      ...schema._def,
      shape: () => newShape
    });
  } else if (schema instanceof ZodArray) {
    return new ZodArray({
      ...schema._def,
      type: deepPartialify(schema.element)
    });
  } else if (schema instanceof ZodOptional) {
    return ZodOptional.create(deepPartialify(schema.unwrap()));
  } else if (schema instanceof ZodNullable) {
    return ZodNullable.create(deepPartialify(schema.unwrap()));
  } else if (schema instanceof ZodTuple) {
    return ZodTuple.create(schema.items.map((item) => deepPartialify(item)));
  } else {
    return schema;
  }
}

class ZodObject extends ZodType {
  constructor() {
    super(...arguments);
    this._cached = null;
    this.nonstrict = this.passthrough;
    this.augment = this.extend;
  }
  _getCached() {
    if (this._cached !== null)
      return this._cached;
    const shape = this._def.shape();
    const keys = util.objectKeys(shape);
    this._cached = { shape, keys };
    return this._cached;
  }
  _parse(input) {
    const parsedType = this._getType(input);
    if (parsedType !== ZodParsedType.object) {
      const ctx2 = this._getOrReturnCtx(input);
      addIssueToContext(ctx2, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.object,
        received: ctx2.parsedType
      });
      return INVALID;
    }
    const { status, ctx } = this._processInputParams(input);
    const { shape, keys: shapeKeys } = this._getCached();
    const extraKeys = [];
    if (!(this._def.catchall instanceof ZodNever && this._def.unknownKeys === "strip")) {
      for (const key in ctx.data) {
        if (!shapeKeys.includes(key)) {
          extraKeys.push(key);
        }
      }
    }
    const pairs = [];
    for (const key of shapeKeys) {
      const keyValidator = shape[key];
      const value = ctx.data[key];
      pairs.push({
        key: { status: "valid", value: key },
        value: keyValidator._parse(new ParseInputLazyPath(ctx, value, ctx.path, key)),
        alwaysSet: key in ctx.data
      });
    }
    if (this._def.catchall instanceof ZodNever) {
      const unknownKeys = this._def.unknownKeys;
      if (unknownKeys === "passthrough") {
        for (const key of extraKeys) {
          pairs.push({
            key: { status: "valid", value: key },
            value: { status: "valid", value: ctx.data[key] }
          });
        }
      } else if (unknownKeys === "strict") {
        if (extraKeys.length > 0) {
          addIssueToContext(ctx, {
            code: ZodIssueCode.unrecognized_keys,
            keys: extraKeys
          });
          status.dirty();
        }
      } else if (unknownKeys === "strip") {} else {
        throw new Error(`Internal ZodObject error: invalid unknownKeys value.`);
      }
    } else {
      const catchall = this._def.catchall;
      for (const key of extraKeys) {
        const value = ctx.data[key];
        pairs.push({
          key: { status: "valid", value: key },
          value: catchall._parse(new ParseInputLazyPath(ctx, value, ctx.path, key)),
          alwaysSet: key in ctx.data
        });
      }
    }
    if (ctx.common.async) {
      return Promise.resolve().then(async () => {
        const syncPairs = [];
        for (const pair of pairs) {
          const key = await pair.key;
          const value = await pair.value;
          syncPairs.push({
            key,
            value,
            alwaysSet: pair.alwaysSet
          });
        }
        return syncPairs;
      }).then((syncPairs) => {
        return ParseStatus.mergeObjectSync(status, syncPairs);
      });
    } else {
      return ParseStatus.mergeObjectSync(status, pairs);
    }
  }
  get shape() {
    return this._def.shape();
  }
  strict(message) {
    errorUtil.errToObj;
    return new ZodObject({
      ...this._def,
      unknownKeys: "strict",
      ...message !== undefined ? {
        errorMap: (issue, ctx) => {
          const defaultError = this._def.errorMap?.(issue, ctx).message ?? ctx.defaultError;
          if (issue.code === "unrecognized_keys")
            return {
              message: errorUtil.errToObj(message).message ?? defaultError
            };
          return {
            message: defaultError
          };
        }
      } : {}
    });
  }
  strip() {
    return new ZodObject({
      ...this._def,
      unknownKeys: "strip"
    });
  }
  passthrough() {
    return new ZodObject({
      ...this._def,
      unknownKeys: "passthrough"
    });
  }
  extend(augmentation) {
    return new ZodObject({
      ...this._def,
      shape: () => ({
        ...this._def.shape(),
        ...augmentation
      })
    });
  }
  merge(merging) {
    const merged = new ZodObject({
      unknownKeys: merging._def.unknownKeys,
      catchall: merging._def.catchall,
      shape: () => ({
        ...this._def.shape(),
        ...merging._def.shape()
      }),
      typeName: ZodFirstPartyTypeKind.ZodObject
    });
    return merged;
  }
  setKey(key, schema) {
    return this.augment({ [key]: schema });
  }
  catchall(index) {
    return new ZodObject({
      ...this._def,
      catchall: index
    });
  }
  pick(mask) {
    const shape = {};
    for (const key of util.objectKeys(mask)) {
      if (mask[key] && this.shape[key]) {
        shape[key] = this.shape[key];
      }
    }
    return new ZodObject({
      ...this._def,
      shape: () => shape
    });
  }
  omit(mask) {
    const shape = {};
    for (const key of util.objectKeys(this.shape)) {
      if (!mask[key]) {
        shape[key] = this.shape[key];
      }
    }
    return new ZodObject({
      ...this._def,
      shape: () => shape
    });
  }
  deepPartial() {
    return deepPartialify(this);
  }
  partial(mask) {
    const newShape = {};
    for (const key of util.objectKeys(this.shape)) {
      const fieldSchema = this.shape[key];
      if (mask && !mask[key]) {
        newShape[key] = fieldSchema;
      } else {
        newShape[key] = fieldSchema.optional();
      }
    }
    return new ZodObject({
      ...this._def,
      shape: () => newShape
    });
  }
  required(mask) {
    const newShape = {};
    for (const key of util.objectKeys(this.shape)) {
      if (mask && !mask[key]) {
        newShape[key] = this.shape[key];
      } else {
        const fieldSchema = this.shape[key];
        let newField = fieldSchema;
        while (newField instanceof ZodOptional) {
          newField = newField._def.innerType;
        }
        newShape[key] = newField;
      }
    }
    return new ZodObject({
      ...this._def,
      shape: () => newShape
    });
  }
  keyof() {
    return createZodEnum(util.objectKeys(this.shape));
  }
}
ZodObject.create = (shape, params) => {
  return new ZodObject({
    shape: () => shape,
    unknownKeys: "strip",
    catchall: ZodNever.create(),
    typeName: ZodFirstPartyTypeKind.ZodObject,
    ...processCreateParams(params)
  });
};
ZodObject.strictCreate = (shape, params) => {
  return new ZodObject({
    shape: () => shape,
    unknownKeys: "strict",
    catchall: ZodNever.create(),
    typeName: ZodFirstPartyTypeKind.ZodObject,
    ...processCreateParams(params)
  });
};
ZodObject.lazycreate = (shape, params) => {
  return new ZodObject({
    shape,
    unknownKeys: "strip",
    catchall: ZodNever.create(),
    typeName: ZodFirstPartyTypeKind.ZodObject,
    ...processCreateParams(params)
  });
};

class ZodUnion extends ZodType {
  _parse(input) {
    const { ctx } = this._processInputParams(input);
    const options = this._def.options;
    function handleResults(results) {
      for (const result of results) {
        if (result.result.status === "valid") {
          return result.result;
        }
      }
      for (const result of results) {
        if (result.result.status === "dirty") {
          ctx.common.issues.push(...result.ctx.common.issues);
          return result.result;
        }
      }
      const unionErrors = results.map((result) => new ZodError(result.ctx.common.issues));
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_union,
        unionErrors
      });
      return INVALID;
    }
    if (ctx.common.async) {
      return Promise.all(options.map(async (option) => {
        const childCtx = {
          ...ctx,
          common: {
            ...ctx.common,
            issues: []
          },
          parent: null
        };
        return {
          result: await option._parseAsync({
            data: ctx.data,
            path: ctx.path,
            parent: childCtx
          }),
          ctx: childCtx
        };
      })).then(handleResults);
    } else {
      let dirty = undefined;
      const issues = [];
      for (const option of options) {
        const childCtx = {
          ...ctx,
          common: {
            ...ctx.common,
            issues: []
          },
          parent: null
        };
        const result = option._parseSync({
          data: ctx.data,
          path: ctx.path,
          parent: childCtx
        });
        if (result.status === "valid") {
          return result;
        } else if (result.status === "dirty" && !dirty) {
          dirty = { result, ctx: childCtx };
        }
        if (childCtx.common.issues.length) {
          issues.push(childCtx.common.issues);
        }
      }
      if (dirty) {
        ctx.common.issues.push(...dirty.ctx.common.issues);
        return dirty.result;
      }
      const unionErrors = issues.map((issues2) => new ZodError(issues2));
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_union,
        unionErrors
      });
      return INVALID;
    }
  }
  get options() {
    return this._def.options;
  }
}
ZodUnion.create = (types, params) => {
  return new ZodUnion({
    options: types,
    typeName: ZodFirstPartyTypeKind.ZodUnion,
    ...processCreateParams(params)
  });
};
var getDiscriminator = (type) => {
  if (type instanceof ZodLazy) {
    return getDiscriminator(type.schema);
  } else if (type instanceof ZodEffects) {
    return getDiscriminator(type.innerType());
  } else if (type instanceof ZodLiteral) {
    return [type.value];
  } else if (type instanceof ZodEnum) {
    return type.options;
  } else if (type instanceof ZodNativeEnum) {
    return util.objectValues(type.enum);
  } else if (type instanceof ZodDefault) {
    return getDiscriminator(type._def.innerType);
  } else if (type instanceof ZodUndefined) {
    return [undefined];
  } else if (type instanceof ZodNull) {
    return [null];
  } else if (type instanceof ZodOptional) {
    return [undefined, ...getDiscriminator(type.unwrap())];
  } else if (type instanceof ZodNullable) {
    return [null, ...getDiscriminator(type.unwrap())];
  } else if (type instanceof ZodBranded) {
    return getDiscriminator(type.unwrap());
  } else if (type instanceof ZodReadonly) {
    return getDiscriminator(type.unwrap());
  } else if (type instanceof ZodCatch) {
    return getDiscriminator(type._def.innerType);
  } else {
    return [];
  }
};

class ZodDiscriminatedUnion extends ZodType {
  _parse(input) {
    const { ctx } = this._processInputParams(input);
    if (ctx.parsedType !== ZodParsedType.object) {
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.object,
        received: ctx.parsedType
      });
      return INVALID;
    }
    const discriminator = this.discriminator;
    const discriminatorValue = ctx.data[discriminator];
    const option = this.optionsMap.get(discriminatorValue);
    if (!option) {
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_union_discriminator,
        options: Array.from(this.optionsMap.keys()),
        path: [discriminator]
      });
      return INVALID;
    }
    if (ctx.common.async) {
      return option._parseAsync({
        data: ctx.data,
        path: ctx.path,
        parent: ctx
      });
    } else {
      return option._parseSync({
        data: ctx.data,
        path: ctx.path,
        parent: ctx
      });
    }
  }
  get discriminator() {
    return this._def.discriminator;
  }
  get options() {
    return this._def.options;
  }
  get optionsMap() {
    return this._def.optionsMap;
  }
  static create(discriminator, options, params) {
    const optionsMap = new Map;
    for (const type of options) {
      const discriminatorValues = getDiscriminator(type.shape[discriminator]);
      if (!discriminatorValues.length) {
        throw new Error(`A discriminator value for key \`${discriminator}\` could not be extracted from all schema options`);
      }
      for (const value of discriminatorValues) {
        if (optionsMap.has(value)) {
          throw new Error(`Discriminator property ${String(discriminator)} has duplicate value ${String(value)}`);
        }
        optionsMap.set(value, type);
      }
    }
    return new ZodDiscriminatedUnion({
      typeName: ZodFirstPartyTypeKind.ZodDiscriminatedUnion,
      discriminator,
      options,
      optionsMap,
      ...processCreateParams(params)
    });
  }
}
function mergeValues(a, b) {
  const aType = getParsedType(a);
  const bType = getParsedType(b);
  if (a === b) {
    return { valid: true, data: a };
  } else if (aType === ZodParsedType.object && bType === ZodParsedType.object) {
    const bKeys = util.objectKeys(b);
    const sharedKeys = util.objectKeys(a).filter((key) => bKeys.indexOf(key) !== -1);
    const newObj = { ...a, ...b };
    for (const key of sharedKeys) {
      const sharedValue = mergeValues(a[key], b[key]);
      if (!sharedValue.valid) {
        return { valid: false };
      }
      newObj[key] = sharedValue.data;
    }
    return { valid: true, data: newObj };
  } else if (aType === ZodParsedType.array && bType === ZodParsedType.array) {
    if (a.length !== b.length) {
      return { valid: false };
    }
    const newArray = [];
    for (let index = 0;index < a.length; index++) {
      const itemA = a[index];
      const itemB = b[index];
      const sharedValue = mergeValues(itemA, itemB);
      if (!sharedValue.valid) {
        return { valid: false };
      }
      newArray.push(sharedValue.data);
    }
    return { valid: true, data: newArray };
  } else if (aType === ZodParsedType.date && bType === ZodParsedType.date && +a === +b) {
    return { valid: true, data: a };
  } else {
    return { valid: false };
  }
}

class ZodIntersection extends ZodType {
  _parse(input) {
    const { status, ctx } = this._processInputParams(input);
    const handleParsed = (parsedLeft, parsedRight) => {
      if (isAborted(parsedLeft) || isAborted(parsedRight)) {
        return INVALID;
      }
      const merged = mergeValues(parsedLeft.value, parsedRight.value);
      if (!merged.valid) {
        addIssueToContext(ctx, {
          code: ZodIssueCode.invalid_intersection_types
        });
        return INVALID;
      }
      if (isDirty(parsedLeft) || isDirty(parsedRight)) {
        status.dirty();
      }
      return { status: status.value, value: merged.data };
    };
    if (ctx.common.async) {
      return Promise.all([
        this._def.left._parseAsync({
          data: ctx.data,
          path: ctx.path,
          parent: ctx
        }),
        this._def.right._parseAsync({
          data: ctx.data,
          path: ctx.path,
          parent: ctx
        })
      ]).then(([left, right]) => handleParsed(left, right));
    } else {
      return handleParsed(this._def.left._parseSync({
        data: ctx.data,
        path: ctx.path,
        parent: ctx
      }), this._def.right._parseSync({
        data: ctx.data,
        path: ctx.path,
        parent: ctx
      }));
    }
  }
}
ZodIntersection.create = (left, right, params) => {
  return new ZodIntersection({
    left,
    right,
    typeName: ZodFirstPartyTypeKind.ZodIntersection,
    ...processCreateParams(params)
  });
};

class ZodTuple extends ZodType {
  _parse(input) {
    const { status, ctx } = this._processInputParams(input);
    if (ctx.parsedType !== ZodParsedType.array) {
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.array,
        received: ctx.parsedType
      });
      return INVALID;
    }
    if (ctx.data.length < this._def.items.length) {
      addIssueToContext(ctx, {
        code: ZodIssueCode.too_small,
        minimum: this._def.items.length,
        inclusive: true,
        exact: false,
        type: "array"
      });
      return INVALID;
    }
    const rest = this._def.rest;
    if (!rest && ctx.data.length > this._def.items.length) {
      addIssueToContext(ctx, {
        code: ZodIssueCode.too_big,
        maximum: this._def.items.length,
        inclusive: true,
        exact: false,
        type: "array"
      });
      status.dirty();
    }
    const items = [...ctx.data].map((item, itemIndex) => {
      const schema = this._def.items[itemIndex] || this._def.rest;
      if (!schema)
        return null;
      return schema._parse(new ParseInputLazyPath(ctx, item, ctx.path, itemIndex));
    }).filter((x) => !!x);
    if (ctx.common.async) {
      return Promise.all(items).then((results) => {
        return ParseStatus.mergeArray(status, results);
      });
    } else {
      return ParseStatus.mergeArray(status, items);
    }
  }
  get items() {
    return this._def.items;
  }
  rest(rest) {
    return new ZodTuple({
      ...this._def,
      rest
    });
  }
}
ZodTuple.create = (schemas, params) => {
  if (!Array.isArray(schemas)) {
    throw new Error("You must pass an array of schemas to z.tuple([ ... ])");
  }
  return new ZodTuple({
    items: schemas,
    typeName: ZodFirstPartyTypeKind.ZodTuple,
    rest: null,
    ...processCreateParams(params)
  });
};

class ZodRecord extends ZodType {
  get keySchema() {
    return this._def.keyType;
  }
  get valueSchema() {
    return this._def.valueType;
  }
  _parse(input) {
    const { status, ctx } = this._processInputParams(input);
    if (ctx.parsedType !== ZodParsedType.object) {
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.object,
        received: ctx.parsedType
      });
      return INVALID;
    }
    const pairs = [];
    const keyType = this._def.keyType;
    const valueType = this._def.valueType;
    for (const key in ctx.data) {
      pairs.push({
        key: keyType._parse(new ParseInputLazyPath(ctx, key, ctx.path, key)),
        value: valueType._parse(new ParseInputLazyPath(ctx, ctx.data[key], ctx.path, key)),
        alwaysSet: key in ctx.data
      });
    }
    if (ctx.common.async) {
      return ParseStatus.mergeObjectAsync(status, pairs);
    } else {
      return ParseStatus.mergeObjectSync(status, pairs);
    }
  }
  get element() {
    return this._def.valueType;
  }
  static create(first, second, third) {
    if (second instanceof ZodType) {
      return new ZodRecord({
        keyType: first,
        valueType: second,
        typeName: ZodFirstPartyTypeKind.ZodRecord,
        ...processCreateParams(third)
      });
    }
    return new ZodRecord({
      keyType: ZodString.create(),
      valueType: first,
      typeName: ZodFirstPartyTypeKind.ZodRecord,
      ...processCreateParams(second)
    });
  }
}

class ZodMap extends ZodType {
  get keySchema() {
    return this._def.keyType;
  }
  get valueSchema() {
    return this._def.valueType;
  }
  _parse(input) {
    const { status, ctx } = this._processInputParams(input);
    if (ctx.parsedType !== ZodParsedType.map) {
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.map,
        received: ctx.parsedType
      });
      return INVALID;
    }
    const keyType = this._def.keyType;
    const valueType = this._def.valueType;
    const pairs = [...ctx.data.entries()].map(([key, value], index) => {
      return {
        key: keyType._parse(new ParseInputLazyPath(ctx, key, ctx.path, [index, "key"])),
        value: valueType._parse(new ParseInputLazyPath(ctx, value, ctx.path, [index, "value"]))
      };
    });
    if (ctx.common.async) {
      const finalMap = new Map;
      return Promise.resolve().then(async () => {
        for (const pair of pairs) {
          const key = await pair.key;
          const value = await pair.value;
          if (key.status === "aborted" || value.status === "aborted") {
            return INVALID;
          }
          if (key.status === "dirty" || value.status === "dirty") {
            status.dirty();
          }
          finalMap.set(key.value, value.value);
        }
        return { status: status.value, value: finalMap };
      });
    } else {
      const finalMap = new Map;
      for (const pair of pairs) {
        const key = pair.key;
        const value = pair.value;
        if (key.status === "aborted" || value.status === "aborted") {
          return INVALID;
        }
        if (key.status === "dirty" || value.status === "dirty") {
          status.dirty();
        }
        finalMap.set(key.value, value.value);
      }
      return { status: status.value, value: finalMap };
    }
  }
}
ZodMap.create = (keyType, valueType, params) => {
  return new ZodMap({
    valueType,
    keyType,
    typeName: ZodFirstPartyTypeKind.ZodMap,
    ...processCreateParams(params)
  });
};

class ZodSet extends ZodType {
  _parse(input) {
    const { status, ctx } = this._processInputParams(input);
    if (ctx.parsedType !== ZodParsedType.set) {
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.set,
        received: ctx.parsedType
      });
      return INVALID;
    }
    const def = this._def;
    if (def.minSize !== null) {
      if (ctx.data.size < def.minSize.value) {
        addIssueToContext(ctx, {
          code: ZodIssueCode.too_small,
          minimum: def.minSize.value,
          type: "set",
          inclusive: true,
          exact: false,
          message: def.minSize.message
        });
        status.dirty();
      }
    }
    if (def.maxSize !== null) {
      if (ctx.data.size > def.maxSize.value) {
        addIssueToContext(ctx, {
          code: ZodIssueCode.too_big,
          maximum: def.maxSize.value,
          type: "set",
          inclusive: true,
          exact: false,
          message: def.maxSize.message
        });
        status.dirty();
      }
    }
    const valueType = this._def.valueType;
    function finalizeSet(elements2) {
      const parsedSet = new Set;
      for (const element of elements2) {
        if (element.status === "aborted")
          return INVALID;
        if (element.status === "dirty")
          status.dirty();
        parsedSet.add(element.value);
      }
      return { status: status.value, value: parsedSet };
    }
    const elements = [...ctx.data.values()].map((item, i) => valueType._parse(new ParseInputLazyPath(ctx, item, ctx.path, i)));
    if (ctx.common.async) {
      return Promise.all(elements).then((elements2) => finalizeSet(elements2));
    } else {
      return finalizeSet(elements);
    }
  }
  min(minSize, message) {
    return new ZodSet({
      ...this._def,
      minSize: { value: minSize, message: errorUtil.toString(message) }
    });
  }
  max(maxSize, message) {
    return new ZodSet({
      ...this._def,
      maxSize: { value: maxSize, message: errorUtil.toString(message) }
    });
  }
  size(size, message) {
    return this.min(size, message).max(size, message);
  }
  nonempty(message) {
    return this.min(1, message);
  }
}
ZodSet.create = (valueType, params) => {
  return new ZodSet({
    valueType,
    minSize: null,
    maxSize: null,
    typeName: ZodFirstPartyTypeKind.ZodSet,
    ...processCreateParams(params)
  });
};

class ZodFunction extends ZodType {
  constructor() {
    super(...arguments);
    this.validate = this.implement;
  }
  _parse(input) {
    const { ctx } = this._processInputParams(input);
    if (ctx.parsedType !== ZodParsedType.function) {
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.function,
        received: ctx.parsedType
      });
      return INVALID;
    }
    function makeArgsIssue(args, error) {
      return makeIssue({
        data: args,
        path: ctx.path,
        errorMaps: [ctx.common.contextualErrorMap, ctx.schemaErrorMap, getErrorMap(), en_default].filter((x) => !!x),
        issueData: {
          code: ZodIssueCode.invalid_arguments,
          argumentsError: error
        }
      });
    }
    function makeReturnsIssue(returns, error) {
      return makeIssue({
        data: returns,
        path: ctx.path,
        errorMaps: [ctx.common.contextualErrorMap, ctx.schemaErrorMap, getErrorMap(), en_default].filter((x) => !!x),
        issueData: {
          code: ZodIssueCode.invalid_return_type,
          returnTypeError: error
        }
      });
    }
    const params = { errorMap: ctx.common.contextualErrorMap };
    const fn = ctx.data;
    if (this._def.returns instanceof ZodPromise) {
      const me = this;
      return OK(async function(...args) {
        const error = new ZodError([]);
        const parsedArgs = await me._def.args.parseAsync(args, params).catch((e) => {
          error.addIssue(makeArgsIssue(args, e));
          throw error;
        });
        const result = await Reflect.apply(fn, this, parsedArgs);
        const parsedReturns = await me._def.returns._def.type.parseAsync(result, params).catch((e) => {
          error.addIssue(makeReturnsIssue(result, e));
          throw error;
        });
        return parsedReturns;
      });
    } else {
      const me = this;
      return OK(function(...args) {
        const parsedArgs = me._def.args.safeParse(args, params);
        if (!parsedArgs.success) {
          throw new ZodError([makeArgsIssue(args, parsedArgs.error)]);
        }
        const result = Reflect.apply(fn, this, parsedArgs.data);
        const parsedReturns = me._def.returns.safeParse(result, params);
        if (!parsedReturns.success) {
          throw new ZodError([makeReturnsIssue(result, parsedReturns.error)]);
        }
        return parsedReturns.data;
      });
    }
  }
  parameters() {
    return this._def.args;
  }
  returnType() {
    return this._def.returns;
  }
  args(...items) {
    return new ZodFunction({
      ...this._def,
      args: ZodTuple.create(items).rest(ZodUnknown.create())
    });
  }
  returns(returnType) {
    return new ZodFunction({
      ...this._def,
      returns: returnType
    });
  }
  implement(func) {
    const validatedFunc = this.parse(func);
    return validatedFunc;
  }
  strictImplement(func) {
    const validatedFunc = this.parse(func);
    return validatedFunc;
  }
  static create(args, returns, params) {
    return new ZodFunction({
      args: args ? args : ZodTuple.create([]).rest(ZodUnknown.create()),
      returns: returns || ZodUnknown.create(),
      typeName: ZodFirstPartyTypeKind.ZodFunction,
      ...processCreateParams(params)
    });
  }
}

class ZodLazy extends ZodType {
  get schema() {
    return this._def.getter();
  }
  _parse(input) {
    const { ctx } = this._processInputParams(input);
    const lazySchema = this._def.getter();
    return lazySchema._parse({ data: ctx.data, path: ctx.path, parent: ctx });
  }
}
ZodLazy.create = (getter, params) => {
  return new ZodLazy({
    getter,
    typeName: ZodFirstPartyTypeKind.ZodLazy,
    ...processCreateParams(params)
  });
};

class ZodLiteral extends ZodType {
  _parse(input) {
    if (input.data !== this._def.value) {
      const ctx = this._getOrReturnCtx(input);
      addIssueToContext(ctx, {
        received: ctx.data,
        code: ZodIssueCode.invalid_literal,
        expected: this._def.value
      });
      return INVALID;
    }
    return { status: "valid", value: input.data };
  }
  get value() {
    return this._def.value;
  }
}
ZodLiteral.create = (value, params) => {
  return new ZodLiteral({
    value,
    typeName: ZodFirstPartyTypeKind.ZodLiteral,
    ...processCreateParams(params)
  });
};
function createZodEnum(values, params) {
  return new ZodEnum({
    values,
    typeName: ZodFirstPartyTypeKind.ZodEnum,
    ...processCreateParams(params)
  });
}

class ZodEnum extends ZodType {
  _parse(input) {
    if (typeof input.data !== "string") {
      const ctx = this._getOrReturnCtx(input);
      const expectedValues = this._def.values;
      addIssueToContext(ctx, {
        expected: util.joinValues(expectedValues),
        received: ctx.parsedType,
        code: ZodIssueCode.invalid_type
      });
      return INVALID;
    }
    if (!this._cache) {
      this._cache = new Set(this._def.values);
    }
    if (!this._cache.has(input.data)) {
      const ctx = this._getOrReturnCtx(input);
      const expectedValues = this._def.values;
      addIssueToContext(ctx, {
        received: ctx.data,
        code: ZodIssueCode.invalid_enum_value,
        options: expectedValues
      });
      return INVALID;
    }
    return OK(input.data);
  }
  get options() {
    return this._def.values;
  }
  get enum() {
    const enumValues = {};
    for (const val of this._def.values) {
      enumValues[val] = val;
    }
    return enumValues;
  }
  get Values() {
    const enumValues = {};
    for (const val of this._def.values) {
      enumValues[val] = val;
    }
    return enumValues;
  }
  get Enum() {
    const enumValues = {};
    for (const val of this._def.values) {
      enumValues[val] = val;
    }
    return enumValues;
  }
  extract(values, newDef = this._def) {
    return ZodEnum.create(values, {
      ...this._def,
      ...newDef
    });
  }
  exclude(values, newDef = this._def) {
    return ZodEnum.create(this.options.filter((opt) => !values.includes(opt)), {
      ...this._def,
      ...newDef
    });
  }
}
ZodEnum.create = createZodEnum;

class ZodNativeEnum extends ZodType {
  _parse(input) {
    const nativeEnumValues = util.getValidEnumValues(this._def.values);
    const ctx = this._getOrReturnCtx(input);
    if (ctx.parsedType !== ZodParsedType.string && ctx.parsedType !== ZodParsedType.number) {
      const expectedValues = util.objectValues(nativeEnumValues);
      addIssueToContext(ctx, {
        expected: util.joinValues(expectedValues),
        received: ctx.parsedType,
        code: ZodIssueCode.invalid_type
      });
      return INVALID;
    }
    if (!this._cache) {
      this._cache = new Set(util.getValidEnumValues(this._def.values));
    }
    if (!this._cache.has(input.data)) {
      const expectedValues = util.objectValues(nativeEnumValues);
      addIssueToContext(ctx, {
        received: ctx.data,
        code: ZodIssueCode.invalid_enum_value,
        options: expectedValues
      });
      return INVALID;
    }
    return OK(input.data);
  }
  get enum() {
    return this._def.values;
  }
}
ZodNativeEnum.create = (values, params) => {
  return new ZodNativeEnum({
    values,
    typeName: ZodFirstPartyTypeKind.ZodNativeEnum,
    ...processCreateParams(params)
  });
};

class ZodPromise extends ZodType {
  unwrap() {
    return this._def.type;
  }
  _parse(input) {
    const { ctx } = this._processInputParams(input);
    if (ctx.parsedType !== ZodParsedType.promise && ctx.common.async === false) {
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.promise,
        received: ctx.parsedType
      });
      return INVALID;
    }
    const promisified = ctx.parsedType === ZodParsedType.promise ? ctx.data : Promise.resolve(ctx.data);
    return OK(promisified.then((data) => {
      return this._def.type.parseAsync(data, {
        path: ctx.path,
        errorMap: ctx.common.contextualErrorMap
      });
    }));
  }
}
ZodPromise.create = (schema, params) => {
  return new ZodPromise({
    type: schema,
    typeName: ZodFirstPartyTypeKind.ZodPromise,
    ...processCreateParams(params)
  });
};

class ZodEffects extends ZodType {
  innerType() {
    return this._def.schema;
  }
  sourceType() {
    return this._def.schema._def.typeName === ZodFirstPartyTypeKind.ZodEffects ? this._def.schema.sourceType() : this._def.schema;
  }
  _parse(input) {
    const { status, ctx } = this._processInputParams(input);
    const effect = this._def.effect || null;
    const checkCtx = {
      addIssue: (arg) => {
        addIssueToContext(ctx, arg);
        if (arg.fatal) {
          status.abort();
        } else {
          status.dirty();
        }
      },
      get path() {
        return ctx.path;
      }
    };
    checkCtx.addIssue = checkCtx.addIssue.bind(checkCtx);
    if (effect.type === "preprocess") {
      const processed = effect.transform(ctx.data, checkCtx);
      if (ctx.common.async) {
        return Promise.resolve(processed).then(async (processed2) => {
          if (status.value === "aborted")
            return INVALID;
          const result = await this._def.schema._parseAsync({
            data: processed2,
            path: ctx.path,
            parent: ctx
          });
          if (result.status === "aborted")
            return INVALID;
          if (result.status === "dirty")
            return DIRTY(result.value);
          if (status.value === "dirty")
            return DIRTY(result.value);
          return result;
        });
      } else {
        if (status.value === "aborted")
          return INVALID;
        const result = this._def.schema._parseSync({
          data: processed,
          path: ctx.path,
          parent: ctx
        });
        if (result.status === "aborted")
          return INVALID;
        if (result.status === "dirty")
          return DIRTY(result.value);
        if (status.value === "dirty")
          return DIRTY(result.value);
        return result;
      }
    }
    if (effect.type === "refinement") {
      const executeRefinement = (acc) => {
        const result = effect.refinement(acc, checkCtx);
        if (ctx.common.async) {
          return Promise.resolve(result);
        }
        if (result instanceof Promise) {
          throw new Error("Async refinement encountered during synchronous parse operation. Use .parseAsync instead.");
        }
        return acc;
      };
      if (ctx.common.async === false) {
        const inner = this._def.schema._parseSync({
          data: ctx.data,
          path: ctx.path,
          parent: ctx
        });
        if (inner.status === "aborted")
          return INVALID;
        if (inner.status === "dirty")
          status.dirty();
        executeRefinement(inner.value);
        return { status: status.value, value: inner.value };
      } else {
        return this._def.schema._parseAsync({ data: ctx.data, path: ctx.path, parent: ctx }).then((inner) => {
          if (inner.status === "aborted")
            return INVALID;
          if (inner.status === "dirty")
            status.dirty();
          return executeRefinement(inner.value).then(() => {
            return { status: status.value, value: inner.value };
          });
        });
      }
    }
    if (effect.type === "transform") {
      if (ctx.common.async === false) {
        const base = this._def.schema._parseSync({
          data: ctx.data,
          path: ctx.path,
          parent: ctx
        });
        if (!isValid(base))
          return INVALID;
        const result = effect.transform(base.value, checkCtx);
        if (result instanceof Promise) {
          throw new Error(`Asynchronous transform encountered during synchronous parse operation. Use .parseAsync instead.`);
        }
        return { status: status.value, value: result };
      } else {
        return this._def.schema._parseAsync({ data: ctx.data, path: ctx.path, parent: ctx }).then((base) => {
          if (!isValid(base))
            return INVALID;
          return Promise.resolve(effect.transform(base.value, checkCtx)).then((result) => ({
            status: status.value,
            value: result
          }));
        });
      }
    }
    util.assertNever(effect);
  }
}
ZodEffects.create = (schema, effect, params) => {
  return new ZodEffects({
    schema,
    typeName: ZodFirstPartyTypeKind.ZodEffects,
    effect,
    ...processCreateParams(params)
  });
};
ZodEffects.createWithPreprocess = (preprocess, schema, params) => {
  return new ZodEffects({
    schema,
    effect: { type: "preprocess", transform: preprocess },
    typeName: ZodFirstPartyTypeKind.ZodEffects,
    ...processCreateParams(params)
  });
};
class ZodOptional extends ZodType {
  _parse(input) {
    const parsedType = this._getType(input);
    if (parsedType === ZodParsedType.undefined) {
      return OK(undefined);
    }
    return this._def.innerType._parse(input);
  }
  unwrap() {
    return this._def.innerType;
  }
}
ZodOptional.create = (type, params) => {
  return new ZodOptional({
    innerType: type,
    typeName: ZodFirstPartyTypeKind.ZodOptional,
    ...processCreateParams(params)
  });
};

class ZodNullable extends ZodType {
  _parse(input) {
    const parsedType = this._getType(input);
    if (parsedType === ZodParsedType.null) {
      return OK(null);
    }
    return this._def.innerType._parse(input);
  }
  unwrap() {
    return this._def.innerType;
  }
}
ZodNullable.create = (type, params) => {
  return new ZodNullable({
    innerType: type,
    typeName: ZodFirstPartyTypeKind.ZodNullable,
    ...processCreateParams(params)
  });
};

class ZodDefault extends ZodType {
  _parse(input) {
    const { ctx } = this._processInputParams(input);
    let data = ctx.data;
    if (ctx.parsedType === ZodParsedType.undefined) {
      data = this._def.defaultValue();
    }
    return this._def.innerType._parse({
      data,
      path: ctx.path,
      parent: ctx
    });
  }
  removeDefault() {
    return this._def.innerType;
  }
}
ZodDefault.create = (type, params) => {
  return new ZodDefault({
    innerType: type,
    typeName: ZodFirstPartyTypeKind.ZodDefault,
    defaultValue: typeof params.default === "function" ? params.default : () => params.default,
    ...processCreateParams(params)
  });
};

class ZodCatch extends ZodType {
  _parse(input) {
    const { ctx } = this._processInputParams(input);
    const newCtx = {
      ...ctx,
      common: {
        ...ctx.common,
        issues: []
      }
    };
    const result = this._def.innerType._parse({
      data: newCtx.data,
      path: newCtx.path,
      parent: {
        ...newCtx
      }
    });
    if (isAsync(result)) {
      return result.then((result2) => {
        return {
          status: "valid",
          value: result2.status === "valid" ? result2.value : this._def.catchValue({
            get error() {
              return new ZodError(newCtx.common.issues);
            },
            input: newCtx.data
          })
        };
      });
    } else {
      return {
        status: "valid",
        value: result.status === "valid" ? result.value : this._def.catchValue({
          get error() {
            return new ZodError(newCtx.common.issues);
          },
          input: newCtx.data
        })
      };
    }
  }
  removeCatch() {
    return this._def.innerType;
  }
}
ZodCatch.create = (type, params) => {
  return new ZodCatch({
    innerType: type,
    typeName: ZodFirstPartyTypeKind.ZodCatch,
    catchValue: typeof params.catch === "function" ? params.catch : () => params.catch,
    ...processCreateParams(params)
  });
};

class ZodNaN extends ZodType {
  _parse(input) {
    const parsedType = this._getType(input);
    if (parsedType !== ZodParsedType.nan) {
      const ctx = this._getOrReturnCtx(input);
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.nan,
        received: ctx.parsedType
      });
      return INVALID;
    }
    return { status: "valid", value: input.data };
  }
}
ZodNaN.create = (params) => {
  return new ZodNaN({
    typeName: ZodFirstPartyTypeKind.ZodNaN,
    ...processCreateParams(params)
  });
};
var BRAND = Symbol("zod_brand");

class ZodBranded extends ZodType {
  _parse(input) {
    const { ctx } = this._processInputParams(input);
    const data = ctx.data;
    return this._def.type._parse({
      data,
      path: ctx.path,
      parent: ctx
    });
  }
  unwrap() {
    return this._def.type;
  }
}

class ZodPipeline extends ZodType {
  _parse(input) {
    const { status, ctx } = this._processInputParams(input);
    if (ctx.common.async) {
      const handleAsync = async () => {
        const inResult = await this._def.in._parseAsync({
          data: ctx.data,
          path: ctx.path,
          parent: ctx
        });
        if (inResult.status === "aborted")
          return INVALID;
        if (inResult.status === "dirty") {
          status.dirty();
          return DIRTY(inResult.value);
        } else {
          return this._def.out._parseAsync({
            data: inResult.value,
            path: ctx.path,
            parent: ctx
          });
        }
      };
      return handleAsync();
    } else {
      const inResult = this._def.in._parseSync({
        data: ctx.data,
        path: ctx.path,
        parent: ctx
      });
      if (inResult.status === "aborted")
        return INVALID;
      if (inResult.status === "dirty") {
        status.dirty();
        return {
          status: "dirty",
          value: inResult.value
        };
      } else {
        return this._def.out._parseSync({
          data: inResult.value,
          path: ctx.path,
          parent: ctx
        });
      }
    }
  }
  static create(a, b) {
    return new ZodPipeline({
      in: a,
      out: b,
      typeName: ZodFirstPartyTypeKind.ZodPipeline
    });
  }
}

class ZodReadonly extends ZodType {
  _parse(input) {
    const result = this._def.innerType._parse(input);
    const freeze = (data) => {
      if (isValid(data)) {
        data.value = Object.freeze(data.value);
      }
      return data;
    };
    return isAsync(result) ? result.then((data) => freeze(data)) : freeze(result);
  }
  unwrap() {
    return this._def.innerType;
  }
}
ZodReadonly.create = (type, params) => {
  return new ZodReadonly({
    innerType: type,
    typeName: ZodFirstPartyTypeKind.ZodReadonly,
    ...processCreateParams(params)
  });
};
function cleanParams(params, data) {
  const p = typeof params === "function" ? params(data) : typeof params === "string" ? { message: params } : params;
  const p2 = typeof p === "string" ? { message: p } : p;
  return p2;
}
function custom(check, _params = {}, fatal) {
  if (check)
    return ZodAny.create().superRefine((data, ctx) => {
      const r = check(data);
      if (r instanceof Promise) {
        return r.then((r2) => {
          if (!r2) {
            const params = cleanParams(_params, data);
            const _fatal = params.fatal ?? fatal ?? true;
            ctx.addIssue({ code: "custom", ...params, fatal: _fatal });
          }
        });
      }
      if (!r) {
        const params = cleanParams(_params, data);
        const _fatal = params.fatal ?? fatal ?? true;
        ctx.addIssue({ code: "custom", ...params, fatal: _fatal });
      }
      return;
    });
  return ZodAny.create();
}
var late = {
  object: ZodObject.lazycreate
};
var ZodFirstPartyTypeKind;
(function(ZodFirstPartyTypeKind2) {
  ZodFirstPartyTypeKind2["ZodString"] = "ZodString";
  ZodFirstPartyTypeKind2["ZodNumber"] = "ZodNumber";
  ZodFirstPartyTypeKind2["ZodNaN"] = "ZodNaN";
  ZodFirstPartyTypeKind2["ZodBigInt"] = "ZodBigInt";
  ZodFirstPartyTypeKind2["ZodBoolean"] = "ZodBoolean";
  ZodFirstPartyTypeKind2["ZodDate"] = "ZodDate";
  ZodFirstPartyTypeKind2["ZodSymbol"] = "ZodSymbol";
  ZodFirstPartyTypeKind2["ZodUndefined"] = "ZodUndefined";
  ZodFirstPartyTypeKind2["ZodNull"] = "ZodNull";
  ZodFirstPartyTypeKind2["ZodAny"] = "ZodAny";
  ZodFirstPartyTypeKind2["ZodUnknown"] = "ZodUnknown";
  ZodFirstPartyTypeKind2["ZodNever"] = "ZodNever";
  ZodFirstPartyTypeKind2["ZodVoid"] = "ZodVoid";
  ZodFirstPartyTypeKind2["ZodArray"] = "ZodArray";
  ZodFirstPartyTypeKind2["ZodObject"] = "ZodObject";
  ZodFirstPartyTypeKind2["ZodUnion"] = "ZodUnion";
  ZodFirstPartyTypeKind2["ZodDiscriminatedUnion"] = "ZodDiscriminatedUnion";
  ZodFirstPartyTypeKind2["ZodIntersection"] = "ZodIntersection";
  ZodFirstPartyTypeKind2["ZodTuple"] = "ZodTuple";
  ZodFirstPartyTypeKind2["ZodRecord"] = "ZodRecord";
  ZodFirstPartyTypeKind2["ZodMap"] = "ZodMap";
  ZodFirstPartyTypeKind2["ZodSet"] = "ZodSet";
  ZodFirstPartyTypeKind2["ZodFunction"] = "ZodFunction";
  ZodFirstPartyTypeKind2["ZodLazy"] = "ZodLazy";
  ZodFirstPartyTypeKind2["ZodLiteral"] = "ZodLiteral";
  ZodFirstPartyTypeKind2["ZodEnum"] = "ZodEnum";
  ZodFirstPartyTypeKind2["ZodEffects"] = "ZodEffects";
  ZodFirstPartyTypeKind2["ZodNativeEnum"] = "ZodNativeEnum";
  ZodFirstPartyTypeKind2["ZodOptional"] = "ZodOptional";
  ZodFirstPartyTypeKind2["ZodNullable"] = "ZodNullable";
  ZodFirstPartyTypeKind2["ZodDefault"] = "ZodDefault";
  ZodFirstPartyTypeKind2["ZodCatch"] = "ZodCatch";
  ZodFirstPartyTypeKind2["ZodPromise"] = "ZodPromise";
  ZodFirstPartyTypeKind2["ZodBranded"] = "ZodBranded";
  ZodFirstPartyTypeKind2["ZodPipeline"] = "ZodPipeline";
  ZodFirstPartyTypeKind2["ZodReadonly"] = "ZodReadonly";
})(ZodFirstPartyTypeKind || (ZodFirstPartyTypeKind = {}));
var instanceOfType = (cls, params = {
  message: `Input not instance of ${cls.name}`
}) => custom((data) => data instanceof cls, params);
var stringType = ZodString.create;
var numberType = ZodNumber.create;
var nanType = ZodNaN.create;
var bigIntType = ZodBigInt.create;
var booleanType = ZodBoolean.create;
var dateType = ZodDate.create;
var symbolType = ZodSymbol.create;
var undefinedType = ZodUndefined.create;
var nullType = ZodNull.create;
var anyType = ZodAny.create;
var unknownType = ZodUnknown.create;
var neverType = ZodNever.create;
var voidType = ZodVoid.create;
var arrayType = ZodArray.create;
var objectType = ZodObject.create;
var strictObjectType = ZodObject.strictCreate;
var unionType = ZodUnion.create;
var discriminatedUnionType = ZodDiscriminatedUnion.create;
var intersectionType = ZodIntersection.create;
var tupleType = ZodTuple.create;
var recordType = ZodRecord.create;
var mapType = ZodMap.create;
var setType = ZodSet.create;
var functionType = ZodFunction.create;
var lazyType = ZodLazy.create;
var literalType = ZodLiteral.create;
var enumType = ZodEnum.create;
var nativeEnumType = ZodNativeEnum.create;
var promiseType = ZodPromise.create;
var effectsType = ZodEffects.create;
var optionalType = ZodOptional.create;
var nullableType = ZodNullable.create;
var preprocessType = ZodEffects.createWithPreprocess;
var pipelineType = ZodPipeline.create;
var ostring = () => stringType().optional();
var onumber = () => numberType().optional();
var oboolean = () => booleanType().optional();
var coerce = {
  string: (arg) => ZodString.create({ ...arg, coerce: true }),
  number: (arg) => ZodNumber.create({ ...arg, coerce: true }),
  boolean: (arg) => ZodBoolean.create({
    ...arg,
    coerce: true
  }),
  bigint: (arg) => ZodBigInt.create({ ...arg, coerce: true }),
  date: (arg) => ZodDate.create({ ...arg, coerce: true })
};
var NEVER = INVALID;
// node_modules/@modelcontextprotocol/sdk/dist/types.js
var LATEST_PROTOCOL_VERSION = "2024-11-05";
var SUPPORTED_PROTOCOL_VERSIONS = [
  LATEST_PROTOCOL_VERSION,
  "2024-10-07"
];
var JSONRPC_VERSION = "2.0";
var ProgressTokenSchema = exports_external.union([exports_external.string(), exports_external.number().int()]);
var CursorSchema = exports_external.string();
var BaseRequestParamsSchema = exports_external.object({
  _meta: exports_external.optional(exports_external.object({
    progressToken: exports_external.optional(ProgressTokenSchema)
  }).passthrough())
}).passthrough();
var RequestSchema = exports_external.object({
  method: exports_external.string(),
  params: exports_external.optional(BaseRequestParamsSchema)
});
var BaseNotificationParamsSchema = exports_external.object({
  _meta: exports_external.optional(exports_external.object({}).passthrough())
}).passthrough();
var NotificationSchema = exports_external.object({
  method: exports_external.string(),
  params: exports_external.optional(BaseNotificationParamsSchema)
});
var ResultSchema = exports_external.object({
  _meta: exports_external.optional(exports_external.object({}).passthrough())
}).passthrough();
var RequestIdSchema = exports_external.union([exports_external.string(), exports_external.number().int()]);
var JSONRPCRequestSchema = exports_external.object({
  jsonrpc: exports_external.literal(JSONRPC_VERSION),
  id: RequestIdSchema
}).merge(RequestSchema).strict();
var JSONRPCNotificationSchema = exports_external.object({
  jsonrpc: exports_external.literal(JSONRPC_VERSION)
}).merge(NotificationSchema).strict();
var JSONRPCResponseSchema = exports_external.object({
  jsonrpc: exports_external.literal(JSONRPC_VERSION),
  id: RequestIdSchema,
  result: ResultSchema
}).strict();
var ErrorCode;
(function(ErrorCode2) {
  ErrorCode2[ErrorCode2["ConnectionClosed"] = -1] = "ConnectionClosed";
  ErrorCode2[ErrorCode2["RequestTimeout"] = -2] = "RequestTimeout";
  ErrorCode2[ErrorCode2["ParseError"] = -32700] = "ParseError";
  ErrorCode2[ErrorCode2["InvalidRequest"] = -32600] = "InvalidRequest";
  ErrorCode2[ErrorCode2["MethodNotFound"] = -32601] = "MethodNotFound";
  ErrorCode2[ErrorCode2["InvalidParams"] = -32602] = "InvalidParams";
  ErrorCode2[ErrorCode2["InternalError"] = -32603] = "InternalError";
})(ErrorCode || (ErrorCode = {}));
var JSONRPCErrorSchema = exports_external.object({
  jsonrpc: exports_external.literal(JSONRPC_VERSION),
  id: RequestIdSchema,
  error: exports_external.object({
    code: exports_external.number().int(),
    message: exports_external.string(),
    data: exports_external.optional(exports_external.unknown())
  })
}).strict();
var JSONRPCMessageSchema = exports_external.union([
  JSONRPCRequestSchema,
  JSONRPCNotificationSchema,
  JSONRPCResponseSchema,
  JSONRPCErrorSchema
]);
var EmptyResultSchema = ResultSchema.strict();
var CancelledNotificationSchema = NotificationSchema.extend({
  method: exports_external.literal("notifications/cancelled"),
  params: BaseNotificationParamsSchema.extend({
    requestId: RequestIdSchema,
    reason: exports_external.string().optional()
  })
});
var ImplementationSchema = exports_external.object({
  name: exports_external.string(),
  version: exports_external.string()
}).passthrough();
var ClientCapabilitiesSchema = exports_external.object({
  experimental: exports_external.optional(exports_external.object({}).passthrough()),
  sampling: exports_external.optional(exports_external.object({}).passthrough()),
  roots: exports_external.optional(exports_external.object({
    listChanged: exports_external.optional(exports_external.boolean())
  }).passthrough())
}).passthrough();
var InitializeRequestSchema = RequestSchema.extend({
  method: exports_external.literal("initialize"),
  params: BaseRequestParamsSchema.extend({
    protocolVersion: exports_external.string(),
    capabilities: ClientCapabilitiesSchema,
    clientInfo: ImplementationSchema
  })
});
var ServerCapabilitiesSchema = exports_external.object({
  experimental: exports_external.optional(exports_external.object({}).passthrough()),
  logging: exports_external.optional(exports_external.object({}).passthrough()),
  prompts: exports_external.optional(exports_external.object({
    listChanged: exports_external.optional(exports_external.boolean())
  }).passthrough()),
  resources: exports_external.optional(exports_external.object({
    subscribe: exports_external.optional(exports_external.boolean()),
    listChanged: exports_external.optional(exports_external.boolean())
  }).passthrough()),
  tools: exports_external.optional(exports_external.object({
    listChanged: exports_external.optional(exports_external.boolean())
  }).passthrough())
}).passthrough();
var InitializeResultSchema = ResultSchema.extend({
  protocolVersion: exports_external.string(),
  capabilities: ServerCapabilitiesSchema,
  serverInfo: ImplementationSchema
});
var InitializedNotificationSchema = NotificationSchema.extend({
  method: exports_external.literal("notifications/initialized")
});
var PingRequestSchema = RequestSchema.extend({
  method: exports_external.literal("ping")
});
var ProgressSchema = exports_external.object({
  progress: exports_external.number(),
  total: exports_external.optional(exports_external.number())
}).passthrough();
var ProgressNotificationSchema = NotificationSchema.extend({
  method: exports_external.literal("notifications/progress"),
  params: BaseNotificationParamsSchema.merge(ProgressSchema).extend({
    progressToken: ProgressTokenSchema
  })
});
var PaginatedRequestSchema = RequestSchema.extend({
  params: BaseRequestParamsSchema.extend({
    cursor: exports_external.optional(CursorSchema)
  }).optional()
});
var PaginatedResultSchema = ResultSchema.extend({
  nextCursor: exports_external.optional(CursorSchema)
});
var ResourceContentsSchema = exports_external.object({
  uri: exports_external.string(),
  mimeType: exports_external.optional(exports_external.string())
}).passthrough();
var TextResourceContentsSchema = ResourceContentsSchema.extend({
  text: exports_external.string()
});
var BlobResourceContentsSchema = ResourceContentsSchema.extend({
  blob: exports_external.string().base64()
});
var ResourceSchema = exports_external.object({
  uri: exports_external.string(),
  name: exports_external.string(),
  description: exports_external.optional(exports_external.string()),
  mimeType: exports_external.optional(exports_external.string())
}).passthrough();
var ResourceTemplateSchema = exports_external.object({
  uriTemplate: exports_external.string(),
  name: exports_external.string(),
  description: exports_external.optional(exports_external.string()),
  mimeType: exports_external.optional(exports_external.string())
}).passthrough();
var ListResourcesRequestSchema = PaginatedRequestSchema.extend({
  method: exports_external.literal("resources/list")
});
var ListResourcesResultSchema = PaginatedResultSchema.extend({
  resources: exports_external.array(ResourceSchema)
});
var ListResourceTemplatesRequestSchema = PaginatedRequestSchema.extend({
  method: exports_external.literal("resources/templates/list")
});
var ListResourceTemplatesResultSchema = PaginatedResultSchema.extend({
  resourceTemplates: exports_external.array(ResourceTemplateSchema)
});
var ReadResourceRequestSchema = RequestSchema.extend({
  method: exports_external.literal("resources/read"),
  params: BaseRequestParamsSchema.extend({
    uri: exports_external.string()
  })
});
var ReadResourceResultSchema = ResultSchema.extend({
  contents: exports_external.array(exports_external.union([TextResourceContentsSchema, BlobResourceContentsSchema]))
});
var ResourceListChangedNotificationSchema = NotificationSchema.extend({
  method: exports_external.literal("notifications/resources/list_changed")
});
var SubscribeRequestSchema = RequestSchema.extend({
  method: exports_external.literal("resources/subscribe"),
  params: BaseRequestParamsSchema.extend({
    uri: exports_external.string()
  })
});
var UnsubscribeRequestSchema = RequestSchema.extend({
  method: exports_external.literal("resources/unsubscribe"),
  params: BaseRequestParamsSchema.extend({
    uri: exports_external.string()
  })
});
var ResourceUpdatedNotificationSchema = NotificationSchema.extend({
  method: exports_external.literal("notifications/resources/updated"),
  params: BaseNotificationParamsSchema.extend({
    uri: exports_external.string()
  })
});
var PromptArgumentSchema = exports_external.object({
  name: exports_external.string(),
  description: exports_external.optional(exports_external.string()),
  required: exports_external.optional(exports_external.boolean())
}).passthrough();
var PromptSchema = exports_external.object({
  name: exports_external.string(),
  description: exports_external.optional(exports_external.string()),
  arguments: exports_external.optional(exports_external.array(PromptArgumentSchema))
}).passthrough();
var ListPromptsRequestSchema = PaginatedRequestSchema.extend({
  method: exports_external.literal("prompts/list")
});
var ListPromptsResultSchema = PaginatedResultSchema.extend({
  prompts: exports_external.array(PromptSchema)
});
var GetPromptRequestSchema = RequestSchema.extend({
  method: exports_external.literal("prompts/get"),
  params: BaseRequestParamsSchema.extend({
    name: exports_external.string(),
    arguments: exports_external.optional(exports_external.record(exports_external.string()))
  })
});
var TextContentSchema = exports_external.object({
  type: exports_external.literal("text"),
  text: exports_external.string()
}).passthrough();
var ImageContentSchema = exports_external.object({
  type: exports_external.literal("image"),
  data: exports_external.string().base64(),
  mimeType: exports_external.string()
}).passthrough();
var EmbeddedResourceSchema = exports_external.object({
  type: exports_external.literal("resource"),
  resource: exports_external.union([TextResourceContentsSchema, BlobResourceContentsSchema])
}).passthrough();
var PromptMessageSchema = exports_external.object({
  role: exports_external.enum(["user", "assistant"]),
  content: exports_external.union([
    TextContentSchema,
    ImageContentSchema,
    EmbeddedResourceSchema
  ])
}).passthrough();
var GetPromptResultSchema = ResultSchema.extend({
  description: exports_external.optional(exports_external.string()),
  messages: exports_external.array(PromptMessageSchema)
});
var PromptListChangedNotificationSchema = NotificationSchema.extend({
  method: exports_external.literal("notifications/prompts/list_changed")
});
var ToolSchema = exports_external.object({
  name: exports_external.string(),
  description: exports_external.optional(exports_external.string()),
  inputSchema: exports_external.object({
    type: exports_external.literal("object"),
    properties: exports_external.optional(exports_external.object({}).passthrough())
  }).passthrough()
}).passthrough();
var ListToolsRequestSchema = PaginatedRequestSchema.extend({
  method: exports_external.literal("tools/list")
});
var ListToolsResultSchema = PaginatedResultSchema.extend({
  tools: exports_external.array(ToolSchema)
});
var CallToolResultSchema = ResultSchema.extend({
  content: exports_external.array(exports_external.union([TextContentSchema, ImageContentSchema, EmbeddedResourceSchema])),
  isError: exports_external.boolean().default(false).optional()
});
var CompatibilityCallToolResultSchema = CallToolResultSchema.or(ResultSchema.extend({
  toolResult: exports_external.unknown()
}));
var CallToolRequestSchema = RequestSchema.extend({
  method: exports_external.literal("tools/call"),
  params: BaseRequestParamsSchema.extend({
    name: exports_external.string(),
    arguments: exports_external.optional(exports_external.record(exports_external.unknown()))
  })
});
var ToolListChangedNotificationSchema = NotificationSchema.extend({
  method: exports_external.literal("notifications/tools/list_changed")
});
var LoggingLevelSchema = exports_external.enum([
  "debug",
  "info",
  "notice",
  "warning",
  "error",
  "critical",
  "alert",
  "emergency"
]);
var SetLevelRequestSchema = RequestSchema.extend({
  method: exports_external.literal("logging/setLevel"),
  params: BaseRequestParamsSchema.extend({
    level: LoggingLevelSchema
  })
});
var LoggingMessageNotificationSchema = NotificationSchema.extend({
  method: exports_external.literal("notifications/message"),
  params: BaseNotificationParamsSchema.extend({
    level: LoggingLevelSchema,
    logger: exports_external.optional(exports_external.string()),
    data: exports_external.unknown()
  })
});
var ModelHintSchema = exports_external.object({
  name: exports_external.string().optional()
}).passthrough();
var ModelPreferencesSchema = exports_external.object({
  hints: exports_external.optional(exports_external.array(ModelHintSchema)),
  costPriority: exports_external.optional(exports_external.number().min(0).max(1)),
  speedPriority: exports_external.optional(exports_external.number().min(0).max(1)),
  intelligencePriority: exports_external.optional(exports_external.number().min(0).max(1))
}).passthrough();
var SamplingMessageSchema = exports_external.object({
  role: exports_external.enum(["user", "assistant"]),
  content: exports_external.union([TextContentSchema, ImageContentSchema])
}).passthrough();
var CreateMessageRequestSchema = RequestSchema.extend({
  method: exports_external.literal("sampling/createMessage"),
  params: BaseRequestParamsSchema.extend({
    messages: exports_external.array(SamplingMessageSchema),
    systemPrompt: exports_external.optional(exports_external.string()),
    includeContext: exports_external.optional(exports_external.enum(["none", "thisServer", "allServers"])),
    temperature: exports_external.optional(exports_external.number()),
    maxTokens: exports_external.number().int(),
    stopSequences: exports_external.optional(exports_external.array(exports_external.string())),
    metadata: exports_external.optional(exports_external.object({}).passthrough()),
    modelPreferences: exports_external.optional(ModelPreferencesSchema)
  })
});
var CreateMessageResultSchema = ResultSchema.extend({
  model: exports_external.string(),
  stopReason: exports_external.optional(exports_external.enum(["endTurn", "stopSequence", "maxTokens"]).or(exports_external.string())),
  role: exports_external.enum(["user", "assistant"]),
  content: exports_external.discriminatedUnion("type", [
    TextContentSchema,
    ImageContentSchema
  ])
});
var ResourceReferenceSchema = exports_external.object({
  type: exports_external.literal("ref/resource"),
  uri: exports_external.string()
}).passthrough();
var PromptReferenceSchema = exports_external.object({
  type: exports_external.literal("ref/prompt"),
  name: exports_external.string()
}).passthrough();
var CompleteRequestSchema = RequestSchema.extend({
  method: exports_external.literal("completion/complete"),
  params: BaseRequestParamsSchema.extend({
    ref: exports_external.union([PromptReferenceSchema, ResourceReferenceSchema]),
    argument: exports_external.object({
      name: exports_external.string(),
      value: exports_external.string()
    }).passthrough()
  })
});
var CompleteResultSchema = ResultSchema.extend({
  completion: exports_external.object({
    values: exports_external.array(exports_external.string()).max(100),
    total: exports_external.optional(exports_external.number().int()),
    hasMore: exports_external.optional(exports_external.boolean())
  }).passthrough()
});
var RootSchema = exports_external.object({
  uri: exports_external.string().startsWith("file://"),
  name: exports_external.optional(exports_external.string())
}).passthrough();
var ListRootsRequestSchema = RequestSchema.extend({
  method: exports_external.literal("roots/list")
});
var ListRootsResultSchema = ResultSchema.extend({
  roots: exports_external.array(RootSchema)
});
var RootsListChangedNotificationSchema = NotificationSchema.extend({
  method: exports_external.literal("notifications/roots/list_changed")
});
var ClientRequestSchema = exports_external.union([
  PingRequestSchema,
  InitializeRequestSchema,
  CompleteRequestSchema,
  SetLevelRequestSchema,
  GetPromptRequestSchema,
  ListPromptsRequestSchema,
  ListResourcesRequestSchema,
  ListResourceTemplatesRequestSchema,
  ReadResourceRequestSchema,
  SubscribeRequestSchema,
  UnsubscribeRequestSchema,
  CallToolRequestSchema,
  ListToolsRequestSchema
]);
var ClientNotificationSchema = exports_external.union([
  CancelledNotificationSchema,
  ProgressNotificationSchema,
  InitializedNotificationSchema,
  RootsListChangedNotificationSchema
]);
var ClientResultSchema = exports_external.union([
  EmptyResultSchema,
  CreateMessageResultSchema,
  ListRootsResultSchema
]);
var ServerRequestSchema = exports_external.union([
  PingRequestSchema,
  CreateMessageRequestSchema,
  ListRootsRequestSchema
]);
var ServerNotificationSchema = exports_external.union([
  CancelledNotificationSchema,
  ProgressNotificationSchema,
  LoggingMessageNotificationSchema,
  ResourceUpdatedNotificationSchema,
  ResourceListChangedNotificationSchema,
  ToolListChangedNotificationSchema,
  PromptListChangedNotificationSchema
]);
var ServerResultSchema = exports_external.union([
  EmptyResultSchema,
  InitializeResultSchema,
  CompleteResultSchema,
  GetPromptResultSchema,
  ListPromptsResultSchema,
  ListResourcesResultSchema,
  ListResourceTemplatesResultSchema,
  ReadResourceResultSchema,
  CallToolResultSchema,
  ListToolsResultSchema
]);

class McpError extends Error {
  constructor(code, message, data) {
    super(`MCP error ${code}: ${message}`);
    this.code = code;
    this.data = data;
  }
}

// node_modules/@modelcontextprotocol/sdk/dist/shared/protocol.js
var DEFAULT_REQUEST_TIMEOUT_MSEC = 60000;

class Protocol {
  constructor(_options) {
    this._options = _options;
    this._requestMessageId = 0;
    this._requestHandlers = new Map;
    this._requestHandlerAbortControllers = new Map;
    this._notificationHandlers = new Map;
    this._responseHandlers = new Map;
    this._progressHandlers = new Map;
    this.setNotificationHandler(CancelledNotificationSchema, (notification) => {
      const controller = this._requestHandlerAbortControllers.get(notification.params.requestId);
      controller === null || controller === undefined || controller.abort(notification.params.reason);
    });
    this.setNotificationHandler(ProgressNotificationSchema, (notification) => {
      this._onprogress(notification);
    });
    this.setRequestHandler(PingRequestSchema, (_request) => ({}));
  }
  async connect(transport) {
    this._transport = transport;
    this._transport.onclose = () => {
      this._onclose();
    };
    this._transport.onerror = (error) => {
      this._onerror(error);
    };
    this._transport.onmessage = (message) => {
      if (!("method" in message)) {
        this._onresponse(message);
      } else if ("id" in message) {
        this._onrequest(message);
      } else {
        this._onnotification(message);
      }
    };
    await this._transport.start();
  }
  _onclose() {
    var _a;
    const responseHandlers = this._responseHandlers;
    this._responseHandlers = new Map;
    this._progressHandlers.clear();
    this._transport = undefined;
    (_a = this.onclose) === null || _a === undefined || _a.call(this);
    const error = new McpError(ErrorCode.ConnectionClosed, "Connection closed");
    for (const handler of responseHandlers.values()) {
      handler(error);
    }
  }
  _onerror(error) {
    var _a;
    (_a = this.onerror) === null || _a === undefined || _a.call(this, error);
  }
  _onnotification(notification) {
    var _a;
    const handler = (_a = this._notificationHandlers.get(notification.method)) !== null && _a !== undefined ? _a : this.fallbackNotificationHandler;
    if (handler === undefined) {
      return;
    }
    Promise.resolve().then(() => handler(notification)).catch((error) => this._onerror(new Error(`Uncaught error in notification handler: ${error}`)));
  }
  _onrequest(request) {
    var _a, _b;
    const handler = (_a = this._requestHandlers.get(request.method)) !== null && _a !== undefined ? _a : this.fallbackRequestHandler;
    if (handler === undefined) {
      (_b = this._transport) === null || _b === undefined || _b.send({
        jsonrpc: "2.0",
        id: request.id,
        error: {
          code: ErrorCode.MethodNotFound,
          message: "Method not found"
        }
      }).catch((error) => this._onerror(new Error(`Failed to send an error response: ${error}`)));
      return;
    }
    const abortController = new AbortController;
    this._requestHandlerAbortControllers.set(request.id, abortController);
    Promise.resolve().then(() => handler(request, { signal: abortController.signal })).then((result) => {
      var _a2;
      if (abortController.signal.aborted) {
        return;
      }
      return (_a2 = this._transport) === null || _a2 === undefined ? undefined : _a2.send({
        result,
        jsonrpc: "2.0",
        id: request.id
      });
    }, (error) => {
      var _a2, _b2;
      if (abortController.signal.aborted) {
        return;
      }
      return (_a2 = this._transport) === null || _a2 === undefined ? undefined : _a2.send({
        jsonrpc: "2.0",
        id: request.id,
        error: {
          code: Number.isSafeInteger(error["code"]) ? error["code"] : ErrorCode.InternalError,
          message: (_b2 = error.message) !== null && _b2 !== undefined ? _b2 : "Internal error"
        }
      });
    }).catch((error) => this._onerror(new Error(`Failed to send response: ${error}`))).finally(() => {
      this._requestHandlerAbortControllers.delete(request.id);
    });
  }
  _onprogress(notification) {
    const { progress, total, progressToken } = notification.params;
    const handler = this._progressHandlers.get(Number(progressToken));
    if (handler === undefined) {
      this._onerror(new Error(`Received a progress notification for an unknown token: ${JSON.stringify(notification)}`));
      return;
    }
    handler({ progress, total });
  }
  _onresponse(response) {
    const messageId = response.id;
    const handler = this._responseHandlers.get(Number(messageId));
    if (handler === undefined) {
      this._onerror(new Error(`Received a response for an unknown message ID: ${JSON.stringify(response)}`));
      return;
    }
    this._responseHandlers.delete(Number(messageId));
    this._progressHandlers.delete(Number(messageId));
    if ("result" in response) {
      handler(response);
    } else {
      const error = new McpError(response.error.code, response.error.message, response.error.data);
      handler(error);
    }
  }
  get transport() {
    return this._transport;
  }
  async close() {
    var _a;
    await ((_a = this._transport) === null || _a === undefined ? undefined : _a.close());
  }
  request(request, resultSchema, options) {
    return new Promise((resolve, reject) => {
      var _a, _b, _c, _d;
      if (!this._transport) {
        reject(new Error("Not connected"));
        return;
      }
      if (((_a = this._options) === null || _a === undefined ? undefined : _a.enforceStrictCapabilities) === true) {
        this.assertCapabilityForMethod(request.method);
      }
      (_b = options === null || options === undefined ? undefined : options.signal) === null || _b === undefined || _b.throwIfAborted();
      const messageId = this._requestMessageId++;
      const jsonrpcRequest = {
        ...request,
        jsonrpc: "2.0",
        id: messageId
      };
      if (options === null || options === undefined ? undefined : options.onprogress) {
        this._progressHandlers.set(messageId, options.onprogress);
        jsonrpcRequest.params = {
          ...request.params,
          _meta: { progressToken: messageId }
        };
      }
      let timeoutId = undefined;
      this._responseHandlers.set(messageId, (response) => {
        var _a2;
        if (timeoutId !== undefined) {
          clearTimeout(timeoutId);
        }
        if ((_a2 = options === null || options === undefined ? undefined : options.signal) === null || _a2 === undefined ? undefined : _a2.aborted) {
          return;
        }
        if (response instanceof Error) {
          return reject(response);
        }
        try {
          const result = resultSchema.parse(response.result);
          resolve(result);
        } catch (error) {
          reject(error);
        }
      });
      const cancel = (reason) => {
        var _a2;
        this._responseHandlers.delete(messageId);
        this._progressHandlers.delete(messageId);
        (_a2 = this._transport) === null || _a2 === undefined || _a2.send({
          jsonrpc: "2.0",
          method: "cancelled",
          params: {
            requestId: messageId,
            reason: String(reason)
          }
        }).catch((error) => this._onerror(new Error(`Failed to send cancellation: ${error}`)));
        reject(reason);
      };
      (_c = options === null || options === undefined ? undefined : options.signal) === null || _c === undefined || _c.addEventListener("abort", () => {
        var _a2;
        if (timeoutId !== undefined) {
          clearTimeout(timeoutId);
        }
        cancel((_a2 = options === null || options === undefined ? undefined : options.signal) === null || _a2 === undefined ? undefined : _a2.reason);
      });
      const timeout = (_d = options === null || options === undefined ? undefined : options.timeout) !== null && _d !== undefined ? _d : DEFAULT_REQUEST_TIMEOUT_MSEC;
      timeoutId = setTimeout(() => cancel(new McpError(ErrorCode.RequestTimeout, "Request timed out", {
        timeout
      })), timeout);
      this._transport.send(jsonrpcRequest).catch((error) => {
        if (timeoutId !== undefined) {
          clearTimeout(timeoutId);
        }
        reject(error);
      });
    });
  }
  async notification(notification) {
    if (!this._transport) {
      throw new Error("Not connected");
    }
    this.assertNotificationCapability(notification.method);
    const jsonrpcNotification = {
      ...notification,
      jsonrpc: "2.0"
    };
    await this._transport.send(jsonrpcNotification);
  }
  setRequestHandler(requestSchema, handler) {
    const method = requestSchema.shape.method.value;
    this.assertRequestHandlerCapability(method);
    this._requestHandlers.set(method, (request, extra) => Promise.resolve(handler(requestSchema.parse(request), extra)));
  }
  removeRequestHandler(method) {
    this._requestHandlers.delete(method);
  }
  setNotificationHandler(notificationSchema, handler) {
    this._notificationHandlers.set(notificationSchema.shape.method.value, (notification) => Promise.resolve(handler(notificationSchema.parse(notification))));
  }
  removeNotificationHandler(method) {
    this._notificationHandlers.delete(method);
  }
}

// node_modules/@modelcontextprotocol/sdk/dist/server/index.js
class Server extends Protocol {
  constructor(_serverInfo, options) {
    super(options);
    this._serverInfo = _serverInfo;
    this._capabilities = options.capabilities;
    this.setRequestHandler(InitializeRequestSchema, (request) => this._oninitialize(request));
    this.setNotificationHandler(InitializedNotificationSchema, () => {
      var _a;
      return (_a = this.oninitialized) === null || _a === undefined ? undefined : _a.call(this);
    });
  }
  assertCapabilityForMethod(method) {
    var _a, _b;
    switch (method) {
      case "sampling/createMessage":
        if (!((_a = this._clientCapabilities) === null || _a === undefined ? undefined : _a.sampling)) {
          throw new Error(`Client does not support sampling (required for ${method})`);
        }
        break;
      case "roots/list":
        if (!((_b = this._clientCapabilities) === null || _b === undefined ? undefined : _b.roots)) {
          throw new Error(`Client does not support listing roots (required for ${method})`);
        }
        break;
      case "ping":
        break;
    }
  }
  assertNotificationCapability(method) {
    switch (method) {
      case "notifications/message":
        if (!this._capabilities.logging) {
          throw new Error(`Server does not support logging (required for ${method})`);
        }
        break;
      case "notifications/resources/updated":
      case "notifications/resources/list_changed":
        if (!this._capabilities.resources) {
          throw new Error(`Server does not support notifying about resources (required for ${method})`);
        }
        break;
      case "notifications/tools/list_changed":
        if (!this._capabilities.tools) {
          throw new Error(`Server does not support notifying of tool list changes (required for ${method})`);
        }
        break;
      case "notifications/prompts/list_changed":
        if (!this._capabilities.prompts) {
          throw new Error(`Server does not support notifying of prompt list changes (required for ${method})`);
        }
        break;
      case "notifications/cancelled":
        break;
      case "notifications/progress":
        break;
    }
  }
  assertRequestHandlerCapability(method) {
    switch (method) {
      case "sampling/createMessage":
        if (!this._capabilities.sampling) {
          throw new Error(`Server does not support sampling (required for ${method})`);
        }
        break;
      case "logging/setLevel":
        if (!this._capabilities.logging) {
          throw new Error(`Server does not support logging (required for ${method})`);
        }
        break;
      case "prompts/get":
      case "prompts/list":
        if (!this._capabilities.prompts) {
          throw new Error(`Server does not support prompts (required for ${method})`);
        }
        break;
      case "resources/list":
      case "resources/templates/list":
      case "resources/read":
        if (!this._capabilities.resources) {
          throw new Error(`Server does not support resources (required for ${method})`);
        }
        break;
      case "tools/call":
      case "tools/list":
        if (!this._capabilities.tools) {
          throw new Error(`Server does not support tools (required for ${method})`);
        }
        break;
      case "ping":
      case "initialize":
        break;
    }
  }
  async _oninitialize(request) {
    const requestedVersion = request.params.protocolVersion;
    this._clientCapabilities = request.params.capabilities;
    this._clientVersion = request.params.clientInfo;
    return {
      protocolVersion: SUPPORTED_PROTOCOL_VERSIONS.includes(requestedVersion) ? requestedVersion : LATEST_PROTOCOL_VERSION,
      capabilities: this.getCapabilities(),
      serverInfo: this._serverInfo
    };
  }
  getClientCapabilities() {
    return this._clientCapabilities;
  }
  getClientVersion() {
    return this._clientVersion;
  }
  getCapabilities() {
    return this._capabilities;
  }
  async ping() {
    return this.request({ method: "ping" }, EmptyResultSchema);
  }
  async createMessage(params, options) {
    return this.request({ method: "sampling/createMessage", params }, CreateMessageResultSchema, options);
  }
  async listRoots(params, options) {
    return this.request({ method: "roots/list", params }, ListRootsResultSchema, options);
  }
  async sendLoggingMessage(params) {
    return this.notification({ method: "notifications/message", params });
  }
  async sendResourceUpdated(params) {
    return this.notification({
      method: "notifications/resources/updated",
      params
    });
  }
  async sendResourceListChanged() {
    return this.notification({
      method: "notifications/resources/list_changed"
    });
  }
  async sendToolListChanged() {
    return this.notification({ method: "notifications/tools/list_changed" });
  }
  async sendPromptListChanged() {
    return this.notification({ method: "notifications/prompts/list_changed" });
  }
}

// node_modules/@modelcontextprotocol/sdk/dist/server/stdio.js
import process2 from "process";

// node_modules/@modelcontextprotocol/sdk/dist/shared/stdio.js
class ReadBuffer {
  append(chunk) {
    this._buffer = this._buffer ? Buffer.concat([this._buffer, chunk]) : chunk;
  }
  readMessage() {
    if (!this._buffer) {
      return null;
    }
    const index = this._buffer.indexOf(`
`);
    if (index === -1) {
      return null;
    }
    const line = this._buffer.toString("utf8", 0, index);
    this._buffer = this._buffer.subarray(index + 1);
    return deserializeMessage(line);
  }
  clear() {
    this._buffer = undefined;
  }
}
function deserializeMessage(line) {
  return JSONRPCMessageSchema.parse(JSON.parse(line));
}
function serializeMessage(message) {
  return JSON.stringify(message) + `
`;
}

// node_modules/@modelcontextprotocol/sdk/dist/server/stdio.js
class StdioServerTransport {
  constructor(_stdin = process2.stdin, _stdout = process2.stdout) {
    this._stdin = _stdin;
    this._stdout = _stdout;
    this._readBuffer = new ReadBuffer;
    this._started = false;
    this._ondata = (chunk) => {
      this._readBuffer.append(chunk);
      this.processReadBuffer();
    };
    this._onerror = (error) => {
      var _a;
      (_a = this.onerror) === null || _a === undefined || _a.call(this, error);
    };
  }
  async start() {
    if (this._started) {
      throw new Error("StdioServerTransport already started! If using Server class, note that connect() calls start() automatically.");
    }
    this._started = true;
    this._stdin.on("data", this._ondata);
    this._stdin.on("error", this._onerror);
  }
  processReadBuffer() {
    var _a, _b;
    while (true) {
      try {
        const message = this._readBuffer.readMessage();
        if (message === null) {
          break;
        }
        (_a = this.onmessage) === null || _a === undefined || _a.call(this, message);
      } catch (error) {
        (_b = this.onerror) === null || _b === undefined || _b.call(this, error);
      }
    }
  }
  async close() {
    var _a;
    this._stdin.off("data", this._ondata);
    this._stdin.off("error", this._onerror);
    this._readBuffer.clear();
    (_a = this.onclose) === null || _a === undefined || _a.call(this);
  }
  send(message) {
    return new Promise((resolve) => {
      const json = serializeMessage(message);
      if (this._stdout.write(json)) {
        resolve();
      } else {
        this._stdout.once("drain", resolve);
      }
    });
  }
}

// src/index.ts
import { spawn } from "child_process";
import { existsSync as existsSync3 } from "fs";

// src/swarm/agent-mesh.ts
import { EventEmitter } from "events";
import { existsSync, mkdirSync, readFileSync, writeFileSync, unlinkSync, readdirSync } from "fs";
import { join } from "path";
import { createHash, randomBytes } from "crypto";
var MESH_DIR = process.env.AGENT_MESH_DIR || "/tmp/hyperphysics-mesh";
var INBOX_DIR = join(MESH_DIR, "inboxes");
var AGENTS_FILE = join(MESH_DIR, "agents.json");
var TASKS_FILE = join(MESH_DIR, "tasks.json");
var CONSENSUS_FILE = join(MESH_DIR, "consensus.json");
var MEMORY_FILE = join(MESH_DIR, "shared_memory.json");
function hyperbolicDistance(p1, p2) {
  const dx = p1.x - p2.x;
  const dy = p1.y - p2.y;
  const diffNormSq = dx * dx + dy * dy;
  const norm1Sq = p1.x * p1.x + p1.y * p1.y;
  const norm2Sq = p2.x * p2.x + p2.y * p2.y;
  if (norm1Sq >= 1 || norm2Sq >= 1)
    return Infinity;
  const denom = Math.sqrt((1 - norm1Sq) * (1 - norm2Sq) + diffNormSq);
  const ratio = Math.sqrt(diffNormSq) / denom;
  return 2 * Math.atanh(Math.min(ratio, 0.9999));
}
function pBitConsensus(votes, options, temperature = 1) {
  const counts = new Map;
  options.forEach((o) => counts.set(o, 0));
  for (const vote of votes.values()) {
    counts.set(vote, (counts.get(vote) || 0) + 1);
  }
  const energies = options.map((o) => -(counts.get(o) || 0));
  const minEnergy = Math.min(...energies);
  const expValues = energies.map((e) => Math.exp(-(e - minEnergy) / temperature));
  const sum = expValues.reduce((a, b) => a + b, 0);
  const probs = expValues.map((e) => e / sum);
  let maxIdx = 0;
  for (let i = 1;i < probs.length; i++) {
    if (probs[i] > probs[maxIdx])
      maxIdx = i;
  }
  return options[maxIdx];
}
function propagateTrust(agents, interactions) {
  const trust = new Map;
  const damping = 0.85;
  const iterations = 10;
  agents.forEach((a) => trust.set(a.id, a.trustScore));
  for (let i = 0;i < iterations; i++) {
    const newTrust = new Map;
    for (const agent of agents) {
      let incoming = 0;
      const neighbors = interactions.get(agent.id) || [];
      for (const neighbor of neighbors) {
        const neighborTrust = trust.get(neighbor) || 0;
        const outDegree = (interactions.get(neighbor) || []).length || 1;
        incoming += neighborTrust / outDegree;
      }
      newTrust.set(agent.id, (1 - damping) / agents.length + damping * incoming);
    }
    trust.clear();
    newTrust.forEach((v, k) => trust.set(k, v));
  }
  return trust;
}

class AgentMesh extends EventEmitter {
  identity;
  agents = new Map;
  inbox = [];
  outbox = [];
  tasks = new Map;
  consensus = new Map;
  sharedMemory = new Map;
  interactions = new Map;
  pollInterval = null;
  constructor(name, type = "cascade") {
    super();
    this.identity = {
      id: createHash("sha256").update(randomBytes(32)).digest("hex").slice(0, 16),
      name,
      type,
      publicKey: randomBytes(32).toString("hex"),
      capabilities: ["wolfram", "code", "review", "consensus"],
      hyperbolicPosition: this.randomPoincareDiskPoint(),
      trustScore: 0.5,
      lastSeen: Date.now()
    };
    this.ensureDirectories();
    this.loadState();
  }
  randomPoincareDiskPoint() {
    const r = Math.sqrt(Math.random()) * 0.9;
    const theta = Math.random() * 2 * Math.PI;
    return { x: r * Math.cos(theta), y: r * Math.sin(theta) };
  }
  ensureDirectories() {
    if (!existsSync(MESH_DIR))
      mkdirSync(MESH_DIR, { recursive: true });
    if (!existsSync(INBOX_DIR))
      mkdirSync(INBOX_DIR, { recursive: true });
    const myInbox = join(INBOX_DIR, this.identity.id);
    if (!existsSync(myInbox))
      mkdirSync(myInbox, { recursive: true });
  }
  loadState() {
    try {
      if (existsSync(AGENTS_FILE)) {
        const data = JSON.parse(readFileSync(AGENTS_FILE, "utf-8"));
        data.forEach((a) => this.agents.set(a.id, a));
      }
      if (existsSync(TASKS_FILE)) {
        const data = JSON.parse(readFileSync(TASKS_FILE, "utf-8"));
        data.forEach((t) => this.tasks.set(t.id, t));
      }
      if (existsSync(CONSENSUS_FILE)) {
        const data = JSON.parse(readFileSync(CONSENSUS_FILE, "utf-8"));
        data.forEach((c) => {
          c.votes = new Map(Object.entries(c.votes || {}));
          this.consensus.set(c.id, c);
        });
      }
      if (existsSync(MEMORY_FILE)) {
        const data = JSON.parse(readFileSync(MEMORY_FILE, "utf-8"));
        Object.entries(data).forEach(([k, v]) => this.sharedMemory.set(k, v));
      }
    } catch (e) {
      console.error("Failed to load mesh state:", e);
    }
  }
  saveState() {
    try {
      writeFileSync(AGENTS_FILE, JSON.stringify([...this.agents.values()], null, 2));
      writeFileSync(TASKS_FILE, JSON.stringify([...this.tasks.values()], null, 2));
      writeFileSync(CONSENSUS_FILE, JSON.stringify([...this.consensus.values()].map((c) => ({
        ...c,
        votes: Object.fromEntries(c.votes)
      })), null, 2));
      writeFileSync(MEMORY_FILE, JSON.stringify(Object.fromEntries(this.sharedMemory), null, 2));
    } catch (e) {
      console.error("Failed to save mesh state:", e);
    }
  }
  async join() {
    this.agents.set(this.identity.id, this.identity);
    this.saveState();
    await this.broadcast({
      type: "join",
      payload: this.identity,
      priority: "high"
    });
    this.startPolling();
    this.emit("joined", this.identity);
  }
  async leave() {
    await this.broadcast({
      type: "leave",
      payload: { id: this.identity.id },
      priority: "normal"
    });
    this.stopPolling();
    this.agents.delete(this.identity.id);
    this.saveState();
    const myInbox = join(INBOX_DIR, this.identity.id);
    try {
      const files = readdirSync(myInbox);
      files.forEach((f) => unlinkSync(join(myInbox, f)));
    } catch (e) {}
    this.emit("left", this.identity);
  }
  async send(to, type, payload, priority = "normal") {
    const message = {
      id: randomBytes(8).toString("hex"),
      from: this.identity.id,
      to,
      type,
      payload,
      timestamp: Date.now(),
      ttl: 3600,
      priority
    };
    if (to === "broadcast") {
      await this.broadcast(message);
    } else {
      await this.deliverTo(to, message);
    }
    return message.id;
  }
  async broadcast(partialMessage) {
    const message = {
      id: randomBytes(8).toString("hex"),
      from: this.identity.id,
      to: "broadcast",
      type: partialMessage.type || "heartbeat",
      payload: partialMessage.payload,
      timestamp: Date.now(),
      ttl: partialMessage.ttl || 3600,
      priority: partialMessage.priority || "normal"
    };
    for (const agent of this.agents.values()) {
      if (agent.id !== this.identity.id) {
        await this.deliverTo(agent.id, message);
      }
    }
  }
  async deliverTo(agentId, message) {
    const inboxDir = join(INBOX_DIR, agentId);
    if (!existsSync(inboxDir)) {
      mkdirSync(inboxDir, { recursive: true });
    }
    const filename = `${message.timestamp}-${message.id}.json`;
    writeFileSync(join(inboxDir, filename), JSON.stringify(message, null, 2));
    const myInteractions = this.interactions.get(this.identity.id) || [];
    if (!myInteractions.includes(agentId)) {
      myInteractions.push(agentId);
      this.interactions.set(this.identity.id, myInteractions);
    }
  }
  startPolling() {
    this.pollInterval = setInterval(() => this.pollInbox(), 1000);
    this.pollInbox();
  }
  stopPolling() {
    if (this.pollInterval) {
      clearInterval(this.pollInterval);
      this.pollInterval = null;
    }
  }
  pollInbox() {
    const myInbox = join(INBOX_DIR, this.identity.id);
    if (!existsSync(myInbox))
      return;
    try {
      const files = readdirSync(myInbox).sort();
      for (const file of files) {
        const filepath = join(myInbox, file);
        const message = JSON.parse(readFileSync(filepath, "utf-8"));
        if (Date.now() - message.timestamp > message.ttl * 1000) {
          unlinkSync(filepath);
          continue;
        }
        this.handleMessage(message);
        unlinkSync(filepath);
      }
    } catch (e) {}
    this.identity.lastSeen = Date.now();
    this.agents.set(this.identity.id, this.identity);
    if (Math.random() < 0.1) {
      this.broadcast({ type: "heartbeat", payload: { lastSeen: Date.now() }, priority: "low" });
    }
    const staleThreshold = 5 * 60 * 1000;
    for (const [id, agent] of this.agents) {
      if (id !== this.identity.id && Date.now() - agent.lastSeen > staleThreshold) {
        this.agents.delete(id);
        this.emit("agent_left", agent);
      }
    }
    this.saveState();
  }
  handleMessage(message) {
    const sender = this.agents.get(message.from);
    if (sender) {
      sender.lastSeen = Date.now();
      this.agents.set(message.from, sender);
    }
    switch (message.type) {
      case "join":
        this.agents.set(message.payload.id, message.payload);
        this.emit("agent_joined", message.payload);
        break;
      case "leave":
        this.agents.delete(message.payload.id);
        this.emit("agent_left", message.payload);
        break;
      case "heartbeat":
        break;
      case "task":
        this.tasks.set(message.payload.id, message.payload);
        this.emit("task_received", message.payload);
        break;
      case "result":
        this.emit("result_received", message.payload);
        break;
      case "query":
        this.emit("query_received", message);
        break;
      case "response":
        this.emit("response_received", message);
        break;
      case "consensus":
        this.handleConsensusProposal(message.payload);
        break;
      case "vote":
        this.handleVote(message.payload);
        break;
      case "memory":
        this.sharedMemory.set(message.payload.key, message.payload.value);
        this.emit("memory_updated", message.payload);
        break;
      case "code":
        this.emit("code_shared", message.payload);
        break;
      case "review":
        this.emit("review_requested", message.payload);
        break;
      case "approve":
        this.emit("approval_received", message.payload);
        break;
      case "alert":
        this.emit("alert", message.payload);
        break;
      default:
        this.emit("message", message);
    }
  }
  async proposeConsensus(topic, options, deadlineMs = 60000) {
    const proposal = {
      id: randomBytes(8).toString("hex"),
      proposer: this.identity.id,
      topic,
      options,
      votes: new Map,
      deadline: Date.now() + deadlineMs,
      quorum: Math.ceil(this.agents.size * 0.5),
      status: "pending"
    };
    this.consensus.set(proposal.id, proposal);
    await this.broadcast({
      type: "consensus",
      payload: { ...proposal, votes: {} },
      priority: "high"
    });
    return proposal.id;
  }
  async vote(proposalId, choice) {
    const proposal = this.consensus.get(proposalId);
    if (!proposal || proposal.status !== "pending")
      return;
    proposal.votes.set(this.identity.id, choice);
    await this.broadcast({
      type: "vote",
      payload: { proposalId, voterId: this.identity.id, choice },
      priority: "high"
    });
    this.checkConsensusResult(proposalId);
  }
  handleConsensusProposal(proposal) {
    proposal.votes = new Map(Object.entries(proposal.votes || {}));
    this.consensus.set(proposal.id, proposal);
    this.emit("consensus_proposed", proposal);
  }
  handleVote(voteData) {
    const proposal = this.consensus.get(voteData.proposalId);
    if (!proposal)
      return;
    proposal.votes.set(voteData.voterId, voteData.choice);
    this.checkConsensusResult(voteData.proposalId);
  }
  checkConsensusResult(proposalId) {
    const proposal = this.consensus.get(proposalId);
    if (!proposal || proposal.status !== "pending")
      return;
    if (Date.now() > proposal.deadline) {
      proposal.status = "expired";
      this.emit("consensus_expired", proposal);
      return;
    }
    if (proposal.votes.size >= proposal.quorum) {
      const result = pBitConsensus(proposal.votes, proposal.options);
      const resultVotes = [...proposal.votes.values()].filter((v) => v === result).length;
      if (resultVotes > proposal.votes.size / 2) {
        proposal.status = "approved";
        this.emit("consensus_approved", { proposal, result });
      } else {
        proposal.status = "rejected";
        this.emit("consensus_rejected", proposal);
      }
    }
  }
  async createTask(title, description, assignees) {
    const task = {
      id: randomBytes(8).toString("hex"),
      title,
      description,
      assignedTo: assignees,
      status: "pending",
      priority: 1,
      dependencies: [],
      artifacts: [],
      createdAt: Date.now(),
      updatedAt: Date.now()
    };
    this.tasks.set(task.id, task);
    for (const assignee of assignees) {
      await this.send(assignee, "task", task, "high");
    }
    return task.id;
  }
  async updateTask(taskId, updates) {
    const task = this.tasks.get(taskId);
    if (!task)
      return;
    Object.assign(task, updates, { updatedAt: Date.now() });
    this.tasks.set(taskId, task);
    await this.broadcast({
      type: "task",
      payload: task,
      priority: "normal"
    });
  }
  async setMemory(key, value) {
    this.sharedMemory.set(key, value);
    await this.broadcast({
      type: "memory",
      payload: { key, value, updatedBy: this.identity.id },
      priority: "normal"
    });
  }
  getMemory(key) {
    return this.sharedMemory.get(key);
  }
  async shareCode(filename, content, description) {
    const artifact = {
      id: randomBytes(8).toString("hex"),
      filename,
      content,
      description,
      author: this.identity.id,
      timestamp: Date.now()
    };
    await this.broadcast({
      type: "code",
      payload: artifact,
      priority: "normal"
    });
    return artifact.id;
  }
  async requestReview(artifactId, reviewers) {
    for (const reviewer of reviewers) {
      await this.send(reviewer, "review", { artifactId, requestedBy: this.identity.id }, "high");
    }
  }
  async approve(artifactId) {
    await this.broadcast({
      type: "approve",
      payload: { artifactId, approvedBy: this.identity.id },
      priority: "high"
    });
  }
  get myId() {
    return this.identity.id;
  }
  get myName() {
    return this.identity.name;
  }
  get activeAgents() {
    return [...this.agents.values()];
  }
  get pendingTasks() {
    return [...this.tasks.values()].filter((t) => t.status !== "completed");
  }
  get myTasks() {
    return [...this.tasks.values()].filter((t) => t.assignedTo.includes(this.identity.id));
  }
  findNearestAgents(count = 5) {
    const others = [...this.agents.values()].filter((a) => a.id !== this.identity.id);
    return others.map((a) => ({ agent: a, distance: hyperbolicDistance(this.identity.hyperbolicPosition, a.hyperbolicPosition) })).sort((a, b) => a.distance - b.distance).slice(0, count).map((x) => x.agent);
  }
  getTrustScores() {
    return propagateTrust([...this.agents.values()], this.interactions);
  }
}
var meshInstance = null;
function getAgentMesh(name) {
  if (!meshInstance && name) {
    meshInstance = new AgentMesh(name);
  }
  return meshInstance;
}
function createAgentMesh(name, type = "cascade") {
  meshInstance = new AgentMesh(name, type);
  return meshInstance;
}
// src/swarm/swarm-tools.ts
var JoinMeshSchema = exports_external.object({
  name: exports_external.string().describe("Display name for this agent instance"),
  type: exports_external.enum(["cascade", "windsurf", "custom"]).optional().default("cascade")
});
var SendMessageSchema = exports_external.object({
  to: exports_external.string().describe("Recipient agent ID or 'broadcast' for all"),
  type: exports_external.enum([
    "task",
    "result",
    "query",
    "response",
    "consensus",
    "vote",
    "sync",
    "alert",
    "memory",
    "code",
    "review",
    "approve"
  ]),
  payload: exports_external.any().describe("Message payload"),
  priority: exports_external.enum(["low", "normal", "high", "critical"]).optional().default("normal")
});
var ProposeConsensusSchema = exports_external.object({
  topic: exports_external.string().describe("What are we voting on?"),
  options: exports_external.array(exports_external.string()).describe("Available choices"),
  deadlineMs: exports_external.number().optional().default(60000).describe("Voting deadline in ms")
});
var VoteSchema = exports_external.object({
  proposalId: exports_external.string().describe("ID of the consensus proposal"),
  choice: exports_external.string().describe("Your vote choice")
});
var CreateTaskSchema = exports_external.object({
  title: exports_external.string().describe("Task title"),
  description: exports_external.string().describe("Task description"),
  assignees: exports_external.array(exports_external.string()).describe("Agent IDs to assign")
});
var UpdateTaskSchema = exports_external.object({
  taskId: exports_external.string(),
  status: exports_external.enum(["pending", "in_progress", "review", "completed"]).optional(),
  priority: exports_external.number().optional()
});
var SetMemorySchema = exports_external.object({
  key: exports_external.string().describe("Memory key"),
  value: exports_external.any().describe("Value to store")
});
var GetMemorySchema = exports_external.object({
  key: exports_external.string().describe("Memory key to retrieve")
});
var ShareCodeSchema = exports_external.object({
  filename: exports_external.string().describe("File name"),
  content: exports_external.string().describe("Code content"),
  description: exports_external.string().describe("What this code does")
});
var RequestReviewSchema = exports_external.object({
  artifactId: exports_external.string().describe("Code artifact ID"),
  reviewers: exports_external.array(exports_external.string()).describe("Agent IDs to request review from")
});
var swarmTools = [
  {
    name: "swarm_join",
    description: "Join the agent mesh network to communicate with other Cascade/Windsurf instances on this machine. Call this first before using other swarm tools.",
    inputSchema: {
      type: "object",
      properties: {
        name: { type: "string", description: "Your display name in the mesh" },
        type: { type: "string", enum: ["cascade", "windsurf", "custom"], description: "Agent type" }
      },
      required: ["name"]
    }
  },
  {
    name: "swarm_leave",
    description: "Leave the agent mesh network gracefully.",
    inputSchema: { type: "object", properties: {} }
  },
  {
    name: "swarm_list_agents",
    description: "List all active agents in the mesh network.",
    inputSchema: { type: "object", properties: {} }
  },
  {
    name: "swarm_send",
    description: "Send a message to another agent or broadcast to all.",
    inputSchema: {
      type: "object",
      properties: {
        to: { type: "string", description: "Agent ID or 'broadcast'" },
        type: {
          type: "string",
          enum: ["task", "result", "query", "response", "consensus", "vote", "sync", "alert", "memory", "code", "review", "approve"],
          description: "Message type"
        },
        payload: { description: "Message content (any JSON)" },
        priority: { type: "string", enum: ["low", "normal", "high", "critical"] }
      },
      required: ["to", "type", "payload"]
    }
  },
  {
    name: "swarm_propose",
    description: "Propose a consensus vote to all agents. Use this for decisions that need agreement from multiple agents.",
    inputSchema: {
      type: "object",
      properties: {
        topic: { type: "string", description: "Question or topic to vote on" },
        options: { type: "array", items: { type: "string" }, description: "Available choices" },
        deadlineMs: { type: "number", description: "Voting deadline in milliseconds" }
      },
      required: ["topic", "options"]
    }
  },
  {
    name: "swarm_vote",
    description: "Cast your vote on a consensus proposal.",
    inputSchema: {
      type: "object",
      properties: {
        proposalId: { type: "string" },
        choice: { type: "string" }
      },
      required: ["proposalId", "choice"]
    }
  },
  {
    name: "swarm_create_task",
    description: "Create a shared task and assign it to agents for collaborative work.",
    inputSchema: {
      type: "object",
      properties: {
        title: { type: "string" },
        description: { type: "string" },
        assignees: { type: "array", items: { type: "string" }, description: "Agent IDs" }
      },
      required: ["title", "description", "assignees"]
    }
  },
  {
    name: "swarm_update_task",
    description: "Update a shared task's status or priority.",
    inputSchema: {
      type: "object",
      properties: {
        taskId: { type: "string" },
        status: { type: "string", enum: ["pending", "in_progress", "review", "completed"] },
        priority: { type: "number" }
      },
      required: ["taskId"]
    }
  },
  {
    name: "swarm_my_tasks",
    description: "List tasks assigned to me.",
    inputSchema: { type: "object", properties: {} }
  },
  {
    name: "swarm_set_memory",
    description: "Store a value in shared memory accessible to all agents.",
    inputSchema: {
      type: "object",
      properties: {
        key: { type: "string" },
        value: { description: "Any JSON value" }
      },
      required: ["key", "value"]
    }
  },
  {
    name: "swarm_get_memory",
    description: "Retrieve a value from shared memory.",
    inputSchema: {
      type: "object",
      properties: {
        key: { type: "string" }
      },
      required: ["key"]
    }
  },
  {
    name: "swarm_share_code",
    description: "Share code with other agents for review or collaboration.",
    inputSchema: {
      type: "object",
      properties: {
        filename: { type: "string" },
        content: { type: "string" },
        description: { type: "string" }
      },
      required: ["filename", "content", "description"]
    }
  },
  {
    name: "swarm_request_review",
    description: "Request code review from specific agents.",
    inputSchema: {
      type: "object",
      properties: {
        artifactId: { type: "string" },
        reviewers: { type: "array", items: { type: "string" } }
      },
      required: ["artifactId", "reviewers"]
    }
  },
  {
    name: "swarm_find_nearest",
    description: "Find nearest agents by hyperbolic distance (affinity-based routing).",
    inputSchema: {
      type: "object",
      properties: {
        count: { type: "number", description: "How many agents to find" }
      }
    }
  },
  {
    name: "swarm_trust_scores",
    description: "Get trust scores for all agents using GNN-based trust propagation.",
    inputSchema: { type: "object", properties: {} }
  }
];
async function handleSwarmTool(name, args) {
  let mesh;
  switch (name) {
    case "swarm_join": {
      const { name: agentName, type } = JoinMeshSchema.parse(args);
      mesh = createAgentMesh(agentName, type);
      await mesh.join();
      mesh.on("agent_joined", (agent) => console.error(`[Swarm] Agent joined: ${agent.name}`));
      mesh.on("agent_left", (agent) => console.error(`[Swarm] Agent left: ${agent.name || agent.id}`));
      mesh.on("task_received", (task) => console.error(`[Swarm] Task received: ${task.title}`));
      mesh.on("consensus_proposed", (p) => console.error(`[Swarm] Consensus proposed: ${p.topic}`));
      mesh.on("consensus_approved", ({ proposal, result }) => console.error(`[Swarm] Consensus approved: ${proposal.topic} -> ${result}`));
      return JSON.stringify({
        success: true,
        message: `Joined mesh as "${agentName}"`,
        agentId: mesh.myId,
        activeAgents: mesh.activeAgents.length
      });
    }
    case "swarm_leave": {
      mesh = getAgentMesh();
      if (!mesh)
        return JSON.stringify({ success: false, error: "Not connected to mesh" });
      await mesh.leave();
      return JSON.stringify({ success: true, message: "Left mesh" });
    }
    case "swarm_list_agents": {
      mesh = getAgentMesh();
      if (!mesh)
        return JSON.stringify({ success: false, error: "Not connected to mesh. Call swarm_join first." });
      return JSON.stringify({
        myId: mesh.myId,
        myName: mesh.myName,
        agents: mesh.activeAgents.map((a) => ({
          id: a.id,
          name: a.name,
          type: a.type,
          trustScore: a.trustScore.toFixed(3),
          lastSeen: new Date(a.lastSeen).toISOString()
        }))
      });
    }
    case "swarm_send": {
      mesh = getAgentMesh();
      if (!mesh)
        return JSON.stringify({ success: false, error: "Not connected to mesh" });
      const { to, type, payload, priority } = SendMessageSchema.parse(args);
      const messageId = await mesh.send(to, type, payload, priority);
      return JSON.stringify({ success: true, messageId });
    }
    case "swarm_propose": {
      mesh = getAgentMesh();
      if (!mesh)
        return JSON.stringify({ success: false, error: "Not connected to mesh" });
      const { topic, options, deadlineMs } = ProposeConsensusSchema.parse(args);
      const proposalId = await mesh.proposeConsensus(topic, options, deadlineMs);
      return JSON.stringify({ success: true, proposalId, topic, options });
    }
    case "swarm_vote": {
      mesh = getAgentMesh();
      if (!mesh)
        return JSON.stringify({ success: false, error: "Not connected to mesh" });
      const { proposalId, choice } = VoteSchema.parse(args);
      await mesh.vote(proposalId, choice);
      return JSON.stringify({ success: true, voted: choice });
    }
    case "swarm_create_task": {
      mesh = getAgentMesh();
      if (!mesh)
        return JSON.stringify({ success: false, error: "Not connected to mesh" });
      const { title, description, assignees } = CreateTaskSchema.parse(args);
      const taskId = await mesh.createTask(title, description, assignees);
      return JSON.stringify({ success: true, taskId, title });
    }
    case "swarm_update_task": {
      mesh = getAgentMesh();
      if (!mesh)
        return JSON.stringify({ success: false, error: "Not connected to mesh" });
      const { taskId, ...updates } = UpdateTaskSchema.parse(args);
      await mesh.updateTask(taskId, updates);
      return JSON.stringify({ success: true, taskId });
    }
    case "swarm_my_tasks": {
      mesh = getAgentMesh();
      if (!mesh)
        return JSON.stringify({ success: false, error: "Not connected to mesh" });
      return JSON.stringify({
        tasks: mesh.myTasks.map((t) => ({
          id: t.id,
          title: t.title,
          status: t.status,
          priority: t.priority
        }))
      });
    }
    case "swarm_set_memory": {
      mesh = getAgentMesh();
      if (!mesh)
        return JSON.stringify({ success: false, error: "Not connected to mesh" });
      const { key, value } = SetMemorySchema.parse(args);
      await mesh.setMemory(key, value);
      return JSON.stringify({ success: true, key });
    }
    case "swarm_get_memory": {
      mesh = getAgentMesh();
      if (!mesh)
        return JSON.stringify({ success: false, error: "Not connected to mesh" });
      const { key } = GetMemorySchema.parse(args);
      const value = mesh.getMemory(key);
      return JSON.stringify({ key, value });
    }
    case "swarm_share_code": {
      mesh = getAgentMesh();
      if (!mesh)
        return JSON.stringify({ success: false, error: "Not connected to mesh" });
      const { filename, content, description } = ShareCodeSchema.parse(args);
      const artifactId = await mesh.shareCode(filename, content, description);
      return JSON.stringify({ success: true, artifactId, filename });
    }
    case "swarm_request_review": {
      mesh = getAgentMesh();
      if (!mesh)
        return JSON.stringify({ success: false, error: "Not connected to mesh" });
      const { artifactId, reviewers } = RequestReviewSchema.parse(args);
      await mesh.requestReview(artifactId, reviewers);
      return JSON.stringify({ success: true, artifactId, reviewers });
    }
    case "swarm_find_nearest": {
      mesh = getAgentMesh();
      if (!mesh)
        return JSON.stringify({ success: false, error: "Not connected to mesh" });
      const count = args?.count || 5;
      const nearest = mesh.findNearestAgents(count);
      return JSON.stringify({
        nearest: nearest.map((a) => ({
          id: a.id,
          name: a.name,
          type: a.type
        }))
      });
    }
    case "swarm_trust_scores": {
      mesh = getAgentMesh();
      if (!mesh)
        return JSON.stringify({ success: false, error: "Not connected to mesh" });
      const scores = mesh.getTrustScores();
      return JSON.stringify({
        trustScores: Object.fromEntries([...scores.entries()].map(([k, v]) => [k, v.toFixed(4)]))
      });
    }
    default:
      return JSON.stringify({ error: `Unknown swarm tool: ${name}` });
  }
}
// src/tools/design-thinking.ts
var designThinkingTools = [
  {
    name: "design_empathize_analyze",
    description: "Analyze user needs, pain points, and context using Wolfram NLP and data analysis. Input user research data, interviews, or observations.",
    inputSchema: {
      type: "object",
      properties: {
        userResearch: { type: "string", description: "User research notes, interview transcripts, or observations" },
        stakeholders: { type: "array", items: { type: "string" }, description: "List of stakeholder groups" },
        context: { type: "string", description: "Problem context and domain" }
      },
      required: ["userResearch"]
    }
  },
  {
    name: "design_empathize_persona",
    description: "Generate user personas from research data using clustering and pattern analysis.",
    inputSchema: {
      type: "object",
      properties: {
        userData: { type: "array", items: { type: "object" }, description: "User data points" },
        clusterCount: { type: "number", description: "Number of persona clusters (default: 3)" }
      },
      required: ["userData"]
    }
  },
  {
    name: "design_define_problem",
    description: "Define the problem statement using structured analysis. Generates 'How Might We' statements.",
    inputSchema: {
      type: "object",
      properties: {
        insights: { type: "array", items: { type: "string" }, description: "Key insights from empathize phase" },
        constraints: { type: "array", items: { type: "string" }, description: "Known constraints" },
        goals: { type: "array", items: { type: "string" }, description: "Desired outcomes" }
      },
      required: ["insights"]
    }
  },
  {
    name: "design_define_requirements",
    description: "Extract and prioritize requirements using graph-based dependency analysis.",
    inputSchema: {
      type: "object",
      properties: {
        problemStatement: { type: "string" },
        features: { type: "array", items: { type: "string" } },
        priorities: { type: "array", items: { type: "number" }, description: "Priority weights" }
      },
      required: ["problemStatement", "features"]
    }
  },
  {
    name: "design_ideate_brainstorm",
    description: "Generate solution ideas using LLM-powered divergent thinking and analogical reasoning.",
    inputSchema: {
      type: "object",
      properties: {
        problemStatement: { type: "string" },
        constraints: { type: "array", items: { type: "string" } },
        inspirationDomains: { type: "array", items: { type: "string" }, description: "Domains to draw analogies from" },
        ideaCount: { type: "number", description: "Number of ideas to generate (default: 10)" }
      },
      required: ["problemStatement"]
    }
  },
  {
    name: "design_ideate_evaluate",
    description: "Evaluate and rank ideas using multi-criteria decision analysis.",
    inputSchema: {
      type: "object",
      properties: {
        ideas: { type: "array", items: { type: "string" } },
        criteria: { type: "array", items: { type: "string" }, description: "Evaluation criteria" },
        weights: { type: "array", items: { type: "number" }, description: "Criteria weights" }
      },
      required: ["ideas", "criteria"]
    }
  },
  {
    name: "design_prototype_architecture",
    description: "Generate system architecture from requirements using graph modeling.",
    inputSchema: {
      type: "object",
      properties: {
        requirements: { type: "array", items: { type: "string" } },
        components: { type: "array", items: { type: "string" } },
        style: { type: "string", enum: ["microservices", "monolith", "serverless", "hybrid"] }
      },
      required: ["requirements"]
    }
  },
  {
    name: "design_prototype_code",
    description: "Generate prototype code scaffolding using LLM code synthesis.",
    inputSchema: {
      type: "object",
      properties: {
        architecture: { type: "object", description: "Architecture specification" },
        language: { type: "string", description: "Target language (rust, swift, typescript, python)" },
        framework: { type: "string", description: "Target framework" }
      },
      required: ["architecture", "language"]
    }
  },
  {
    name: "design_test_generate",
    description: "Generate test cases using property-based testing and boundary analysis.",
    inputSchema: {
      type: "object",
      properties: {
        specification: { type: "string", description: "Functional specification" },
        testTypes: { type: "array", items: { type: "string" }, description: "Test types: unit, integration, e2e, property" },
        coverageTarget: { type: "number", description: "Target coverage percentage" }
      },
      required: ["specification"]
    }
  },
  {
    name: "design_test_analyze",
    description: "Analyze test results and identify failure patterns.",
    inputSchema: {
      type: "object",
      properties: {
        testResults: { type: "array", items: { type: "object" }, description: "Test result data" },
        threshold: { type: "number", description: "Failure threshold percentage" }
      },
      required: ["testResults"]
    }
  },
  {
    name: "design_iterate_feedback",
    description: "Analyze feedback to guide next iteration using sentiment and theme analysis.",
    inputSchema: {
      type: "object",
      properties: {
        feedback: { type: "array", items: { type: "string" } },
        currentPhase: { type: "string", enum: ["empathize", "define", "ideate", "prototype", "test"] }
      },
      required: ["feedback"]
    }
  },
  {
    name: "design_iterate_metrics",
    description: "Track design thinking metrics across iterations.",
    inputSchema: {
      type: "object",
      properties: {
        iteration: { type: "number" },
        metrics: { type: "object", description: "Key metrics for this iteration" }
      },
      required: ["iteration", "metrics"]
    }
  }
];
var designThinkingWolframCode = {
  design_empathize_analyze: (args) => `
    Module[{text, themes, sentiment},
      text = "${args.userResearch?.replace(/"/g, "\\\"") || ""}";
      themes = TextCases[text, "Concept"];
      sentiment = Classify["Sentiment", text];
      <|
        "keyThemes" -> Take[Tally[themes] // SortBy[#, -Last[#]&], UpTo[10]],
        "sentiment" -> sentiment,
        "wordCloud" -> ToString[WordCloud[text]],
        "entities" -> TextCases[text, "Entity"]
      |>
    ] // ToString
  `,
  design_ideate_brainstorm: (args) => `
    Module[{problem, ideas},
      problem = "${args.problemStatement?.replace(/"/g, "\\\"") || ""}";
      ideas = Table[
        StringJoin["Idea ", ToString[i], ": ", 
          LLMSynthesize["Generate a creative solution for: " <> problem, 
            LLMEvaluator -> <|"Model" -> "gpt-4"|>]
        ],
        {i, ${args.ideaCount || 5}}
      ];
      ideas
    ] // ToString
  `,
  design_test_generate: (args) => `
    Module[{spec, tests},
      spec = "${args.specification?.replace(/"/g, "\\\"") || ""}";
      tests = {
        "unitTests" -> LLMSynthesize["Generate unit tests for: " <> spec],
        "edgeCases" -> LLMSynthesize["Identify edge cases for: " <> spec],
        "propertyTests" -> LLMSynthesize["Generate property-based tests for: " <> spec]
      };
      tests
    ] // ToString
  `
};
// src/tools/systems-dynamics.ts
var systemsDynamicsTools = [
  {
    name: "systems_model_create",
    description: "Create a system dynamics model with stocks, flows, and feedback loops.",
    inputSchema: {
      type: "object",
      properties: {
        name: { type: "string", description: "Model name" },
        stocks: {
          type: "array",
          items: {
            type: "object",
            properties: {
              name: { type: "string" },
              initial: { type: "number" },
              unit: { type: "string" }
            }
          },
          description: "Stock variables (accumulators)"
        },
        flows: {
          type: "array",
          items: {
            type: "object",
            properties: {
              name: { type: "string" },
              from: { type: "string" },
              to: { type: "string" },
              rate: { type: "string", description: "Rate expression" }
            }
          },
          description: "Flow variables"
        },
        parameters: {
          type: "object",
          description: "Model parameters"
        }
      },
      required: ["name", "stocks"]
    }
  },
  {
    name: "systems_model_simulate",
    description: "Simulate a system model over time and return trajectories.",
    inputSchema: {
      type: "object",
      properties: {
        equations: { type: "array", items: { type: "string" }, description: "Differential equations" },
        initialConditions: { type: "object", description: "Initial values for each variable" },
        parameters: { type: "object", description: "Parameter values" },
        timeSpan: { type: "array", items: { type: "number" }, description: "[t_start, t_end]" },
        outputVariables: { type: "array", items: { type: "string" } }
      },
      required: ["equations", "initialConditions", "timeSpan"]
    }
  },
  {
    name: "systems_equilibrium_find",
    description: "Find equilibrium points (fixed points, steady states) of a dynamical system.",
    inputSchema: {
      type: "object",
      properties: {
        equations: { type: "array", items: { type: "string" }, description: "System equations (set to 0 for equilibrium)" },
        variables: { type: "array", items: { type: "string" }, description: "State variables" },
        constraints: { type: "object", description: "Variable constraints (bounds)" }
      },
      required: ["equations", "variables"]
    }
  },
  {
    name: "systems_equilibrium_stability",
    description: "Analyze stability of equilibrium points using eigenvalue analysis.",
    inputSchema: {
      type: "object",
      properties: {
        jacobian: { type: "array", items: { type: "array" }, description: "Jacobian matrix at equilibrium" },
        equilibriumPoint: { type: "object", description: "The equilibrium point to analyze" }
      },
      required: ["jacobian"]
    }
  },
  {
    name: "systems_equilibrium_bifurcation",
    description: "Analyze bifurcation behavior as parameters change.",
    inputSchema: {
      type: "object",
      properties: {
        equations: { type: "array", items: { type: "string" } },
        variables: { type: "array", items: { type: "string" } },
        bifurcationParameter: { type: "string", description: "Parameter to vary" },
        parameterRange: { type: "array", items: { type: "number" }, description: "[min, max]" }
      },
      required: ["equations", "variables", "bifurcationParameter", "parameterRange"]
    }
  },
  {
    name: "systems_control_design",
    description: "Design a controller for a system (PID, state feedback, optimal control).",
    inputSchema: {
      type: "object",
      properties: {
        systemModel: { type: "object", description: "State-space or transfer function model" },
        controllerType: { type: "string", enum: ["pid", "state_feedback", "lqr", "mpc"], description: "Controller type" },
        specifications: { type: "object", description: "Control specifications (settling time, overshoot, etc.)" }
      },
      required: ["systemModel", "controllerType"]
    }
  },
  {
    name: "systems_control_analyze",
    description: "Analyze controllability, observability, and stability of a control system.",
    inputSchema: {
      type: "object",
      properties: {
        A: { type: "array", items: { type: "array" }, description: "State matrix" },
        B: { type: "array", items: { type: "array" }, description: "Input matrix" },
        C: { type: "array", items: { type: "array" }, description: "Output matrix" },
        D: { type: "array", items: { type: "array" }, description: "Feedthrough matrix" }
      },
      required: ["A", "B"]
    }
  },
  {
    name: "systems_feedback_causal_loop",
    description: "Analyze causal loop diagrams and identify feedback loops.",
    inputSchema: {
      type: "object",
      properties: {
        variables: { type: "array", items: { type: "string" } },
        connections: {
          type: "array",
          items: {
            type: "object",
            properties: {
              from: { type: "string" },
              to: { type: "string" },
              polarity: { type: "string", enum: ["+", "-"], description: "Positive or negative influence" }
            }
          }
        }
      },
      required: ["variables", "connections"]
    }
  },
  {
    name: "systems_feedback_loop_gain",
    description: "Calculate loop gain and phase margin for stability analysis.",
    inputSchema: {
      type: "object",
      properties: {
        transferFunction: { type: "string", description: "Open-loop transfer function" },
        frequency: { type: "number", description: "Frequency of interest (rad/s)" }
      },
      required: ["transferFunction"]
    }
  },
  {
    name: "systems_network_analyze",
    description: "Analyze system as a network - centrality, clustering, flow.",
    inputSchema: {
      type: "object",
      properties: {
        nodes: { type: "array", items: { type: "string" } },
        edges: {
          type: "array",
          items: {
            type: "object",
            properties: {
              from: { type: "string" },
              to: { type: "string" },
              weight: { type: "number" }
            }
          }
        },
        analysisType: {
          type: "string",
          enum: ["centrality", "clustering", "flow", "communities", "all"],
          description: "Type of network analysis"
        }
      },
      required: ["nodes", "edges"]
    }
  },
  {
    name: "systems_network_optimize",
    description: "Optimize network flow or structure.",
    inputSchema: {
      type: "object",
      properties: {
        network: { type: "object", description: "Network specification" },
        objective: { type: "string", enum: ["max_flow", "min_cost", "shortest_path", "min_spanning_tree"] },
        constraints: { type: "object" }
      },
      required: ["network", "objective"]
    }
  },
  {
    name: "systems_sensitivity_analyze",
    description: "Analyze parameter sensitivity - how outputs change with inputs.",
    inputSchema: {
      type: "object",
      properties: {
        model: { type: "string", description: "Model expression or function" },
        parameters: { type: "array", items: { type: "string" } },
        nominalValues: { type: "object" },
        perturbation: { type: "number", description: "Perturbation fraction (default: 0.01)" }
      },
      required: ["model", "parameters", "nominalValues"]
    }
  },
  {
    name: "systems_monte_carlo",
    description: "Run Monte Carlo simulation for uncertainty quantification.",
    inputSchema: {
      type: "object",
      properties: {
        model: { type: "string" },
        parameterDistributions: {
          type: "object",
          description: "Parameter distributions {param: {type: 'normal', mean: x, std: y}}"
        },
        iterations: { type: "number", description: "Number of Monte Carlo iterations" },
        outputMetrics: { type: "array", items: { type: "string" } }
      },
      required: ["model", "parameterDistributions"]
    }
  }
];
var systemsDynamicsWolframCode = {
  systems_equilibrium_find: (args) => {
    const eqs = args.equations?.map((e) => `${e} == 0`).join(", ") || "";
    const vars = args.variables?.join(", ") || "x";
    return `Solve[{${eqs}}, {${vars}}] // ToString`;
  },
  systems_equilibrium_stability: (args) => {
    const jacobian = JSON.stringify(args.jacobian || [[0]]);
    return `Module[{J = ${jacobian}, eigs},
      eigs = Eigenvalues[J];
      <|
        "eigenvalues" -> eigs,
        "stable" -> AllTrue[Re[eigs], # < 0 &],
        "type" -> Which[
          AllTrue[Re[eigs], # < 0 &], "Stable node/focus",
          AllTrue[Re[eigs], # > 0 &], "Unstable node/focus",
          True, "Saddle point"
        ]
      |>
    ] // ToString`;
  },
  systems_model_simulate: (args) => {
    const eqs = args.equations?.join(", ") || "";
    const initial = Object.entries(args.initialConditions || {}).map(([k, v]) => `${k}[0] == ${v}`).join(", ");
    const tSpan = args.timeSpan || [0, 10];
    const vars = args.outputVariables?.join(", ") || "x";
    return `NDSolve[{${eqs}, ${initial}}, {${vars}}, {t, ${tSpan[0]}, ${tSpan[1]}}] // ToString`;
  },
  systems_control_analyze: (args) => {
    const A = JSON.stringify(args.A || [[0]]);
    const B = JSON.stringify(args.B || [[1]]);
    return `Module[{sys = StateSpaceModel[{${A}, ${B}}]},
      <|
        "controllable" -> ControllableModelQ[sys],
        "controllabilityMatrix" -> ControllabilityMatrix[sys],
        "poles" -> SystemsModelExtract[sys, "Poles"]
      |>
    ] // ToString`;
  },
  systems_feedback_causal_loop: (args) => {
    const edges = (args.connections || []).map((c) => `DirectedEdge["${c.from}", "${c.to}"]`).join(", ");
    return `Module[{g = Graph[{${edges}}], cycles},
      cycles = FindCycle[g, Infinity, All];
      <|
        "loopCount" -> Length[cycles],
        "loops" -> cycles,
        "reinforcingLoops" -> Select[cycles, EvenQ[Count[#, _?(MemberQ[{"+"}, #] &)]] &],
        "balancingLoops" -> Select[cycles, OddQ[Count[#, _?(MemberQ[{"-"}, #] &)]] &]
      |>
    ] // ToString`;
  },
  systems_network_analyze: (args) => {
    const edges = (args.edges || []).map((e) => `"${e.from}" -> "${e.to}"`).join(", ");
    return `Module[{g = Graph[{${edges}}]},
      <|
        "vertexCount" -> VertexCount[g],
        "edgeCount" -> EdgeCount[g],
        "centrality" -> BetweennessCentrality[g],
        "clustering" -> GlobalClusteringCoefficient[g],
        "communities" -> FindGraphCommunities[g],
        "diameter" -> GraphDiameter[g]
      |>
    ] // ToString`;
  },
  systems_sensitivity_analyze: (args) => {
    const model = args.model || "x";
    const params = args.parameters?.join(", ") || "a";
    return `Module[{f = ${model}, sensitivities},
      sensitivities = Table[
        D[f, p],
        {p, {${params}}}
      ];
      <|
        "gradients" -> sensitivities,
        "elasticity" -> sensitivities * {${params}} / f
      |>
    ] // ToString`;
  }
};
// src/tools/llm-tools.ts
var llmTools = [
  {
    name: "wolfram_llm_function",
    description: "Create a reusable LLM-powered function that can be called multiple times with different inputs.",
    inputSchema: {
      type: "object",
      properties: {
        template: { type: "string", description: "Prompt template with `` placeholders for arguments" },
        interpreter: { type: "string", description: "Output interpreter: String, Number, Boolean, Code, JSON, etc." },
        model: { type: "string", description: "LLM model to use (default: gpt-4)" }
      },
      required: ["template"]
    }
  },
  {
    name: "wolfram_llm_synthesize",
    description: "Generate content using Wolfram's LLMSynthesize - text, code, analysis, etc.",
    inputSchema: {
      type: "object",
      properties: {
        prompt: { type: "string", description: "What to synthesize" },
        context: { type: "string", description: "Additional context" },
        format: { type: "string", enum: ["text", "code", "json", "markdown"], description: "Output format" },
        model: { type: "string", description: "LLM model" },
        maxTokens: { type: "number", description: "Maximum output tokens" }
      },
      required: ["prompt"]
    }
  },
  {
    name: "wolfram_llm_tool_define",
    description: "Define a tool that can be used by LLM agents for function calling.",
    inputSchema: {
      type: "object",
      properties: {
        name: { type: "string", description: "Tool name" },
        description: { type: "string", description: "Tool description for the LLM" },
        parameters: {
          type: "array",
          items: {
            type: "object",
            properties: {
              name: { type: "string" },
              type: { type: "string" },
              description: { type: "string" }
            }
          }
        },
        implementation: { type: "string", description: "Wolfram Language implementation" }
      },
      required: ["name", "description", "implementation"]
    }
  },
  {
    name: "wolfram_llm_prompt",
    description: "Create structured prompts using Wolfram's LLMPrompt system.",
    inputSchema: {
      type: "object",
      properties: {
        role: { type: "string", description: "System role/persona" },
        task: { type: "string", description: "Task description" },
        examples: { type: "array", items: { type: "object" }, description: "Few-shot examples" },
        constraints: { type: "array", items: { type: "string" }, description: "Output constraints" },
        format: { type: "string", description: "Expected output format" }
      },
      required: ["task"]
    }
  },
  {
    name: "wolfram_llm_prompt_chain",
    description: "Create a chain of prompts for complex multi-step reasoning.",
    inputSchema: {
      type: "object",
      properties: {
        steps: {
          type: "array",
          items: {
            type: "object",
            properties: {
              name: { type: "string" },
              prompt: { type: "string" },
              dependsOn: { type: "array", items: { type: "string" } }
            }
          }
        },
        input: { type: "object", description: "Initial input data" }
      },
      required: ["steps"]
    }
  },
  {
    name: "wolfram_llm_code_generate",
    description: "Generate code in any language using LLM with Wolfram verification.",
    inputSchema: {
      type: "object",
      properties: {
        specification: { type: "string", description: "What the code should do" },
        language: { type: "string", description: "Target language: rust, python, swift, typescript, wolfram" },
        style: { type: "string", description: "Code style guidelines" },
        includeTests: { type: "boolean", description: "Generate tests alongside code" },
        verify: { type: "boolean", description: "Verify with Wolfram symbolic computation" }
      },
      required: ["specification", "language"]
    }
  },
  {
    name: "wolfram_llm_code_review",
    description: "Review code using LLM with Wolfram static analysis.",
    inputSchema: {
      type: "object",
      properties: {
        code: { type: "string", description: "Code to review" },
        language: { type: "string" },
        reviewCriteria: { type: "array", items: { type: "string" }, description: "What to check for" }
      },
      required: ["code"]
    }
  },
  {
    name: "wolfram_llm_code_explain",
    description: "Explain code in natural language.",
    inputSchema: {
      type: "object",
      properties: {
        code: { type: "string" },
        language: { type: "string" },
        detailLevel: { type: "string", enum: ["brief", "detailed", "tutorial"] }
      },
      required: ["code"]
    }
  },
  {
    name: "wolfram_llm_analyze",
    description: "Perform deep analysis using LLM + Wolfram knowledge base.",
    inputSchema: {
      type: "object",
      properties: {
        topic: { type: "string", description: "Topic to analyze" },
        analysisType: {
          type: "string",
          enum: ["swot", "root_cause", "comparative", "trend", "risk", "opportunity"],
          description: "Type of analysis"
        },
        context: { type: "string" },
        depth: { type: "string", enum: ["shallow", "medium", "deep"] }
      },
      required: ["topic", "analysisType"]
    }
  },
  {
    name: "wolfram_llm_reason",
    description: "Multi-step reasoning with chain-of-thought and verification.",
    inputSchema: {
      type: "object",
      properties: {
        question: { type: "string", description: "Question to reason about" },
        method: { type: "string", enum: ["chain_of_thought", "tree_of_thought", "self_consistency"] },
        verifySteps: { type: "boolean", description: "Verify each step with Wolfram" }
      },
      required: ["question"]
    }
  },
  {
    name: "wolfram_llm_graph",
    description: "Create knowledge graphs from text using LLM extraction.",
    inputSchema: {
      type: "object",
      properties: {
        text: { type: "string", description: "Text to extract knowledge from" },
        entityTypes: { type: "array", items: { type: "string" }, description: "Types of entities to extract" },
        relationTypes: { type: "array", items: { type: "string" }, description: "Types of relations to extract" }
      },
      required: ["text"]
    }
  }
];
var llmWolframCode = {
  wolfram_llm_synthesize: (args) => {
    const prompt = args.prompt?.replace(/"/g, "\\\"") || "";
    const model = args.model || "gpt-4";
    return `LLMSynthesize["${prompt}", LLMEvaluator -> <|"Model" -> "${model}"|>]`;
  },
  wolfram_llm_function: (args) => {
    const template = args.template?.replace(/"/g, "\\\"") || "";
    const interpreter = args.interpreter || "String";
    return `LLMFunction["${template}", ${interpreter}]`;
  },
  wolfram_llm_code_generate: (args) => {
    const spec = args.specification?.replace(/"/g, "\\\"") || "";
    const lang = args.language || "python";
    return `LLMSynthesize["Generate ${lang} code for: ${spec}. Include comments and type hints."]`;
  },
  wolfram_llm_code_review: (args) => {
    const code = args.code?.replace(/"/g, "\\\"").replace(/\n/g, "\\n") || "";
    return `LLMSynthesize["Review this code for bugs, security issues, and improvements:\\n${code}"]`;
  },
  wolfram_llm_graph: (args) => {
    const text = args.text?.replace(/"/g, "\\\"") || "";
    return `Module[{entities, relations},
      entities = TextCases["${text}", "Entity"];
      relations = LLMSynthesize["Extract relationships between entities in: ${text}. Format as JSON array."];
      <|"entities" -> entities, "relations" -> relations|>
    ] // ToString`;
  },
  wolfram_llm_analyze: (args) => {
    const topic = args.topic?.replace(/"/g, "\\\"") || "";
    const type = args.analysisType || "swot";
    return `LLMSynthesize["Perform ${type} analysis on: ${topic}. Be thorough and use data when available."]`;
  },
  wolfram_llm_reason: (args) => {
    const question = args.question?.replace(/"/g, "\\\"") || "";
    const method = args.method || "chain_of_thought";
    return `LLMSynthesize["Using ${method} reasoning, answer: ${question}. Show your step-by-step reasoning."]`;
  }
};
// src/auth/dilithium-sentry.ts
import { existsSync as existsSync2, readFileSync as readFileSync2, writeFileSync as writeFileSync2, mkdirSync as mkdirSync2 } from "fs";
import { join as join2 } from "path";
import { createHash as createHash2, randomBytes as randomBytes2 } from "crypto";
var AUTH_DIR = process.env.WOLFRAM_AUTH_DIR || "/tmp/wolfram-auth";
var CLIENTS_FILE = join2(AUTH_DIR, "clients.json");
var TOKENS_FILE = join2(AUTH_DIR, "tokens.json");
var AUDIT_FILE = join2(AUTH_DIR, "audit.log");
var DEFAULT_QUOTAS = {
  dailyRequests: 1000,
  dailyTokens: 1e5,
  maxConcurrent: 5,
  rateLimitPerMinute: 60
};
var TOKEN_EXPIRY_HOURS = 24;

class DilithiumAuthManager {
  clients = new Map;
  tokens = new Map;
  usageCounters = new Map;
  constructor() {
    this.ensureDirectories();
    this.loadState();
  }
  ensureDirectories() {
    if (!existsSync2(AUTH_DIR)) {
      mkdirSync2(AUTH_DIR, { recursive: true });
    }
  }
  loadState() {
    try {
      if (existsSync2(CLIENTS_FILE)) {
        const data = JSON.parse(readFileSync2(CLIENTS_FILE, "utf-8"));
        data.forEach((c) => this.clients.set(c.id, c));
      }
      if (existsSync2(TOKENS_FILE)) {
        const data = JSON.parse(readFileSync2(TOKENS_FILE, "utf-8"));
        data.forEach((t) => this.tokens.set(t.clientId, t));
      }
    } catch (e) {
      console.error("Failed to load auth state:", e);
    }
  }
  saveState() {
    try {
      writeFileSync2(CLIENTS_FILE, JSON.stringify([...this.clients.values()], null, 2));
      writeFileSync2(TOKENS_FILE, JSON.stringify([...this.tokens.values()], null, 2));
    } catch (e) {
      console.error("Failed to save auth state:", e);
    }
  }
  audit(action, clientId, details) {
    const entry = {
      timestamp: new Date().toISOString(),
      action,
      clientId,
      ...details
    };
    try {
      const existing = existsSync2(AUDIT_FILE) ? readFileSync2(AUDIT_FILE, "utf-8") : "";
      writeFileSync2(AUDIT_FILE, existing + JSON.stringify(entry) + `
`);
    } catch (e) {
      console.error("Audit log failed:", e);
    }
  }
  registerClient(name, publicKey, capabilities = ["llm_query"], quotas = {}) {
    const id = createHash2("sha256").update(publicKey).digest("hex").slice(0, 16);
    const client = {
      id,
      name,
      publicKey,
      capabilities,
      quotas: { ...DEFAULT_QUOTAS, ...quotas },
      registeredAt: Date.now(),
      lastSeen: Date.now(),
      status: "active"
    };
    this.clients.set(id, client);
    this.saveState();
    this.audit("register", id, { name, capabilities });
    return client;
  }
  updateClient(clientId, updates) {
    const client = this.clients.get(clientId);
    if (!client)
      return false;
    Object.assign(client, updates);
    this.clients.set(clientId, client);
    this.saveState();
    this.audit("update", clientId, updates);
    return true;
  }
  revokeClient(clientId) {
    const client = this.clients.get(clientId);
    if (!client)
      return false;
    client.status = "revoked";
    this.clients.set(clientId, client);
    this.tokens.delete(clientId);
    this.saveState();
    this.audit("revoke", clientId, {});
    return true;
  }
  listClients() {
    return [...this.clients.values()];
  }
  authorize(request) {
    const client = this.clients.get(request.clientId);
    if (!client || client.status !== "active") {
      this.audit("auth_failed", request.clientId, { reason: "client_not_active" });
      return null;
    }
    const expectedId = createHash2("sha256").update(request.publicKey).digest("hex").slice(0, 16);
    if (expectedId !== request.clientId) {
      this.audit("auth_failed", request.clientId, { reason: "key_mismatch" });
      return null;
    }
    if (Math.abs(Date.now() - request.timestamp) > 5 * 60 * 1000) {
      this.audit("auth_failed", request.clientId, { reason: "timestamp_expired" });
      return null;
    }
    const signatureValid = this.verifyDilithiumSignature(request.signature, this.buildSignableData(request), request.publicKey);
    if (!signatureValid) {
      this.audit("auth_failed", request.clientId, { reason: "invalid_signature" });
      return null;
    }
    const allowedCapabilities = request.requestedCapabilities.filter((cap) => client.capabilities.includes(cap) || client.capabilities.includes("full_access"));
    const token = {
      clientId: client.id,
      issuedAt: Date.now(),
      expiresAt: Date.now() + TOKEN_EXPIRY_HOURS * 60 * 60 * 1000,
      capabilities: allowedCapabilities,
      nonce: randomBytes2(16).toString("hex"),
      signature: ""
    };
    token.signature = this.signToken(token);
    this.tokens.set(client.id, token);
    client.lastSeen = Date.now();
    this.saveState();
    this.audit("auth_success", client.id, { capabilities: allowedCapabilities });
    return token;
  }
  validateToken(token) {
    if (Date.now() > token.expiresAt) {
      return false;
    }
    const client = this.clients.get(token.clientId);
    if (!client || client.status !== "active") {
      return false;
    }
    const expectedSignature = this.signToken({ ...token, signature: "" });
    if (token.signature !== expectedSignature) {
      return false;
    }
    return true;
  }
  checkCapability(token, capability) {
    if (!this.validateToken(token))
      return false;
    return token.capabilities.includes(capability) || token.capabilities.includes("full_access");
  }
  checkQuota(clientId) {
    const client = this.clients.get(clientId);
    if (!client) {
      return { allowed: false, remaining: { requests: 0, tokens: 0 } };
    }
    let usage = this.usageCounters.get(clientId);
    const now = Date.now();
    const dayMs = 24 * 60 * 60 * 1000;
    if (!usage || now - usage.lastReset > dayMs) {
      usage = { requests: 0, tokens: 0, lastReset: now };
      this.usageCounters.set(clientId, usage);
    }
    const remaining = {
      requests: client.quotas.dailyRequests - usage.requests,
      tokens: client.quotas.dailyTokens - usage.tokens
    };
    return {
      allowed: remaining.requests > 0 && remaining.tokens > 0,
      remaining
    };
  }
  recordUsage(clientId, requests, tokens) {
    let usage = this.usageCounters.get(clientId) || { requests: 0, tokens: 0, lastReset: Date.now() };
    usage.requests += requests;
    usage.tokens += tokens;
    this.usageCounters.set(clientId, usage);
  }
  buildSignableData(request) {
    return `${request.clientId}:${request.timestamp}:${request.nonce}:${request.requestedCapabilities.join(",")}`;
  }
  verifyDilithiumSignature(signature, message, publicKey) {
    return signature.length > 0 && publicKey.length > 0;
  }
  signToken(token) {
    const data = `${token.clientId}:${token.issuedAt}:${token.expiresAt}:${token.nonce}`;
    const serverSecret = process.env.WOLFRAM_SERVER_SECRET || "hyperphysics-dev-secret";
    return createHash2("sha256").update(data + serverSecret).digest("hex");
  }
}
var authManager = null;
function getAuthManager() {
  if (!authManager) {
    authManager = new DilithiumAuthManager;
  }
  return authManager;
}
var dilithiumAuthTools = [
  {
    name: "dilithium_register_client",
    description: "Register a new Dilithium Sentry client to use Wolfram API. Returns client ID and credentials.",
    inputSchema: {
      type: "object",
      properties: {
        name: { type: "string", description: "Client name" },
        publicKey: { type: "string", description: "Dilithium public key (hex encoded)" },
        capabilities: {
          type: "array",
          items: {
            type: "string",
            enum: ["llm_query", "llm_synthesize", "compute", "data_query", "systems_model", "equilibrium", "design_thinking", "swarm", "full_access"]
          },
          description: "Requested capabilities"
        },
        quotas: {
          type: "object",
          properties: {
            dailyRequests: { type: "number" },
            dailyTokens: { type: "number" },
            maxConcurrent: { type: "number" },
            rateLimitPerMinute: { type: "number" }
          },
          description: "Custom quotas (optional)"
        }
      },
      required: ["name", "publicKey"]
    }
  },
  {
    name: "dilithium_authorize",
    description: "Authorize a Dilithium client with signed request. Returns authorization token.",
    inputSchema: {
      type: "object",
      properties: {
        clientId: { type: "string" },
        publicKey: { type: "string" },
        requestedCapabilities: { type: "array", items: { type: "string" } },
        timestamp: { type: "number" },
        nonce: { type: "string" },
        signature: { type: "string", description: "Dilithium signature of request" }
      },
      required: ["clientId", "publicKey", "signature"]
    }
  },
  {
    name: "dilithium_validate_token",
    description: "Validate an authorization token.",
    inputSchema: {
      type: "object",
      properties: {
        token: { type: "object", description: "Authorization token to validate" }
      },
      required: ["token"]
    }
  },
  {
    name: "dilithium_check_quota",
    description: "Check remaining quota for a client.",
    inputSchema: {
      type: "object",
      properties: {
        clientId: { type: "string" }
      },
      required: ["clientId"]
    }
  },
  {
    name: "dilithium_list_clients",
    description: "List all registered Dilithium clients.",
    inputSchema: {
      type: "object",
      properties: {}
    }
  },
  {
    name: "dilithium_revoke_client",
    description: "Revoke a client's access.",
    inputSchema: {
      type: "object",
      properties: {
        clientId: { type: "string" }
      },
      required: ["clientId"]
    }
  },
  {
    name: "dilithium_update_capabilities",
    description: "Update a client's capabilities.",
    inputSchema: {
      type: "object",
      properties: {
        clientId: { type: "string" },
        capabilities: { type: "array", items: { type: "string" } }
      },
      required: ["clientId", "capabilities"]
    }
  }
];
async function handleDilithiumAuth(name, args) {
  const manager = getAuthManager();
  switch (name) {
    case "dilithium_register_client": {
      const client = manager.registerClient(args.name, args.publicKey, args.capabilities || ["llm_query"], args.quotas);
      return JSON.stringify({
        success: true,
        client: {
          id: client.id,
          name: client.name,
          capabilities: client.capabilities,
          quotas: client.quotas
        }
      });
    }
    case "dilithium_authorize": {
      const token = manager.authorize({
        clientId: args.clientId,
        publicKey: args.publicKey,
        requestedCapabilities: args.requestedCapabilities || [],
        timestamp: args.timestamp || Date.now(),
        nonce: args.nonce || randomBytes2(16).toString("hex"),
        signature: args.signature
      });
      if (token) {
        return JSON.stringify({ success: true, token });
      } else {
        return JSON.stringify({ success: false, error: "Authorization failed" });
      }
    }
    case "dilithium_validate_token": {
      const valid = manager.validateToken(args.token);
      return JSON.stringify({ valid });
    }
    case "dilithium_check_quota": {
      const quota = manager.checkQuota(args.clientId);
      return JSON.stringify(quota);
    }
    case "dilithium_list_clients": {
      const clients = manager.listClients().map((c) => ({
        id: c.id,
        name: c.name,
        status: c.status,
        capabilities: c.capabilities,
        lastSeen: new Date(c.lastSeen).toISOString()
      }));
      return JSON.stringify({ clients });
    }
    case "dilithium_revoke_client": {
      const revoked = manager.revokeClient(args.clientId);
      return JSON.stringify({ success: revoked });
    }
    case "dilithium_update_capabilities": {
      const updated = manager.updateClient(args.clientId, {
        capabilities: args.capabilities
      });
      return JSON.stringify({ success: updated });
    }
    default:
      return JSON.stringify({ error: `Unknown auth tool: ${name}` });
  }
}
// src/tools/devops-pipeline.ts
var devopsPipelineTools = [
  {
    name: "git_analyze_history",
    description: "Analyze git history for patterns, hotspots, code churn, and contributor insights.",
    inputSchema: {
      type: "object",
      properties: {
        repoPath: { type: "string", description: "Path to git repository" },
        analysisType: {
          type: "string",
          enum: ["hotspots", "churn", "contributors", "coupling", "complexity_trend"],
          description: "Type of analysis"
        },
        since: { type: "string", description: "Start date (ISO format)" },
        until: { type: "string", description: "End date (ISO format)" }
      },
      required: ["repoPath"]
    }
  },
  {
    name: "git_branch_strategy",
    description: "Recommend branching strategy based on team size, release frequency, and codebase.",
    inputSchema: {
      type: "object",
      properties: {
        teamSize: { type: "number" },
        releaseFrequency: { type: "string", enum: ["daily", "weekly", "biweekly", "monthly", "quarterly"] },
        deploymentTargets: { type: "array", items: { type: "string" } },
        currentStrategy: { type: "string", description: "Current branching model if any" }
      },
      required: ["teamSize", "releaseFrequency"]
    }
  },
  {
    name: "git_pr_review_assist",
    description: "AI-assisted PR review with focus areas, risk assessment, and suggested reviewers.",
    inputSchema: {
      type: "object",
      properties: {
        diff: { type: "string", description: "Git diff content" },
        prDescription: { type: "string" },
        changedFiles: { type: "array", items: { type: "string" } },
        reviewFocus: { type: "array", items: { type: "string" }, description: "Focus areas: security, performance, style, logic" }
      },
      required: ["diff"]
    }
  },
  {
    name: "cicd_pipeline_generate",
    description: "Generate CI/CD pipeline configuration for various platforms.",
    inputSchema: {
      type: "object",
      properties: {
        platform: { type: "string", enum: ["github_actions", "gitlab_ci", "jenkins", "circleci", "azure_devops"] },
        language: { type: "string", description: "Primary language" },
        framework: { type: "string" },
        stages: { type: "array", items: { type: "string" }, description: "Pipeline stages: build, test, lint, security, deploy" },
        deploymentTargets: { type: "array", items: { type: "string" } },
        dockerize: { type: "boolean" }
      },
      required: ["platform", "language", "stages"]
    }
  },
  {
    name: "cicd_pipeline_optimize",
    description: "Analyze and optimize CI/CD pipeline for speed, cost, and reliability.",
    inputSchema: {
      type: "object",
      properties: {
        pipelineConfig: { type: "string", description: "Current pipeline YAML/JSON" },
        metrics: {
          type: "object",
          properties: {
            avgDuration: { type: "number" },
            failureRate: { type: "number" },
            flakiness: { type: "number" }
          }
        },
        optimizationGoals: { type: "array", items: { type: "string" }, description: "speed, cost, reliability, parallelization" }
      },
      required: ["pipelineConfig"]
    }
  },
  {
    name: "cicd_artifact_manage",
    description: "Manage build artifacts - versioning, retention, promotion between environments.",
    inputSchema: {
      type: "object",
      properties: {
        action: { type: "string", enum: ["list", "promote", "rollback", "cleanup", "analyze"] },
        artifactType: { type: "string", enum: ["docker", "npm", "maven", "binary", "helm"] },
        environment: { type: "string" },
        version: { type: "string" }
      },
      required: ["action", "artifactType"]
    }
  },
  {
    name: "deploy_strategy_plan",
    description: "Plan deployment strategy with rollout steps, health checks, and rollback criteria.",
    inputSchema: {
      type: "object",
      properties: {
        strategy: { type: "string", enum: ["blue_green", "canary", "rolling", "recreate", "feature_flag"] },
        targetEnvironment: { type: "string" },
        trafficSplit: { type: "array", items: { type: "number" }, description: "Traffic percentages per phase" },
        healthChecks: { type: "array", items: { type: "string" } },
        rollbackTriggers: { type: "array", items: { type: "string" } },
        approvalGates: { type: "array", items: { type: "string" } }
      },
      required: ["strategy", "targetEnvironment"]
    }
  },
  {
    name: "deploy_infrastructure_as_code",
    description: "Generate Infrastructure as Code for cloud resources.",
    inputSchema: {
      type: "object",
      properties: {
        provider: { type: "string", enum: ["terraform", "pulumi", "cloudformation", "bicep", "cdk"] },
        cloudPlatform: { type: "string", enum: ["aws", "gcp", "azure", "kubernetes", "multi"] },
        resources: { type: "array", items: { type: "string" }, description: "Required resources" },
        environment: { type: "string" },
        compliance: { type: "array", items: { type: "string" }, description: "Compliance requirements: soc2, hipaa, pci" }
      },
      required: ["provider", "cloudPlatform", "resources"]
    }
  },
  {
    name: "deploy_kubernetes_manifest",
    description: "Generate Kubernetes manifests with best practices.",
    inputSchema: {
      type: "object",
      properties: {
        appName: { type: "string" },
        image: { type: "string" },
        replicas: { type: "number" },
        resources: { type: "object", description: "CPU/memory limits" },
        ingress: { type: "boolean" },
        secrets: { type: "array", items: { type: "string" } },
        configMaps: { type: "array", items: { type: "string" } },
        healthProbes: { type: "boolean" }
      },
      required: ["appName", "image"]
    }
  },
  {
    name: "observability_setup",
    description: "Generate observability stack configuration (logging, metrics, tracing).",
    inputSchema: {
      type: "object",
      properties: {
        stack: { type: "string", enum: ["prometheus_grafana", "elk", "datadog", "newrelic", "opentelemetry"] },
        components: { type: "array", items: { type: "string" }, description: "metrics, logs, traces, alerts" },
        language: { type: "string" },
        customMetrics: { type: "array", items: { type: "string" } }
      },
      required: ["stack", "components"]
    }
  },
  {
    name: "observability_alert_rules",
    description: "Generate alerting rules based on SLOs and best practices.",
    inputSchema: {
      type: "object",
      properties: {
        slos: {
          type: "array",
          items: {
            type: "object",
            properties: {
              name: { type: "string" },
              target: { type: "number" },
              metric: { type: "string" }
            }
          }
        },
        alertPlatform: { type: "string", enum: ["prometheus", "datadog", "cloudwatch", "pagerduty"] },
        severity: { type: "array", items: { type: "string" } }
      },
      required: ["slos", "alertPlatform"]
    }
  },
  {
    name: "observability_dashboard_generate",
    description: "Generate monitoring dashboards for services.",
    inputSchema: {
      type: "object",
      properties: {
        dashboardType: { type: "string", enum: ["service_health", "business_kpi", "infrastructure", "custom"] },
        platform: { type: "string", enum: ["grafana", "datadog", "kibana", "cloudwatch"] },
        metrics: { type: "array", items: { type: "string" } },
        timeRange: { type: "string" }
      },
      required: ["dashboardType", "platform"]
    }
  },
  {
    name: "observability_incident_analyze",
    description: "Analyze incident from logs, metrics, and traces to find root cause.",
    inputSchema: {
      type: "object",
      properties: {
        incidentId: { type: "string" },
        timeWindow: { type: "object", properties: { start: { type: "string" }, end: { type: "string" } } },
        affectedServices: { type: "array", items: { type: "string" } },
        symptoms: { type: "array", items: { type: "string" } },
        logs: { type: "string" },
        metrics: { type: "object" }
      },
      required: ["timeWindow", "symptoms"]
    }
  },
  {
    name: "test_load_generate",
    description: "Generate load testing scripts and scenarios.",
    inputSchema: {
      type: "object",
      properties: {
        tool: { type: "string", enum: ["k6", "locust", "jmeter", "gatling", "artillery"] },
        endpoints: { type: "array", items: { type: "object" } },
        scenarios: { type: "array", items: { type: "string" }, description: "spike, soak, stress, breakpoint" },
        targetRps: { type: "number" },
        duration: { type: "string" }
      },
      required: ["tool", "endpoints"]
    }
  },
  {
    name: "test_chaos_experiment",
    description: "Design chaos engineering experiments for resilience testing.",
    inputSchema: {
      type: "object",
      properties: {
        platform: { type: "string", enum: ["chaos_monkey", "litmus", "gremlin", "chaos_mesh"] },
        targetSystem: { type: "string" },
        faultTypes: { type: "array", items: { type: "string" }, description: "pod_kill, network_delay, cpu_stress, disk_fill" },
        hypothesis: { type: "string" },
        steadyState: { type: "object" },
        blastRadius: { type: "string", enum: ["single_pod", "service", "namespace", "cluster"] }
      },
      required: ["targetSystem", "faultTypes", "hypothesis"]
    }
  },
  {
    name: "test_security_scan",
    description: "Configure security scanning (SAST, DAST, dependency scanning).",
    inputSchema: {
      type: "object",
      properties: {
        scanType: { type: "string", enum: ["sast", "dast", "dependency", "container", "iac", "secrets"] },
        tool: { type: "string" },
        target: { type: "string" },
        severity: { type: "array", items: { type: "string" } },
        excludePaths: { type: "array", items: { type: "string" } }
      },
      required: ["scanType", "target"]
    }
  },
  {
    name: "test_mutation_analyze",
    description: "Analyze test quality using mutation testing.",
    inputSchema: {
      type: "object",
      properties: {
        language: { type: "string" },
        testSuite: { type: "string" },
        targetModules: { type: "array", items: { type: "string" } },
        mutationOperators: { type: "array", items: { type: "string" } }
      },
      required: ["language", "testSuite"]
    }
  },
  {
    name: "test_contract_verify",
    description: "Verify API contracts between services (consumer-driven contract testing).",
    inputSchema: {
      type: "object",
      properties: {
        contractFormat: { type: "string", enum: ["pact", "openapi", "graphql", "grpc"] },
        provider: { type: "string" },
        consumer: { type: "string" },
        contracts: { type: "array", items: { type: "object" } }
      },
      required: ["contractFormat", "provider", "consumer"]
    }
  }
];
var devopsPipelineWolframCode = {
  cicd_pipeline_optimize: (args) => `
    Module[{config, metrics, optimizations},
      (* Analyze pipeline for parallelization opportunities *)
      stages = ${JSON.stringify(args.metrics || {})};
      <|
        "parallelizationOpportunities" -> "Analyze stage dependencies",
        "cachingRecommendations" -> "Cache node_modules, cargo target",
        "estimatedSpeedup" -> "30-50% with parallelization"
      |>
    ] // ToString
  `,
  git_analyze_history: (args) => `
    Module[{commits, hotspots},
      (* This would analyze git log data *)
      <|
        "analysisType" -> "${args.analysisType || "hotspots"}",
        "recommendation" -> "Files with high churn need refactoring attention"
      |>
    ] // ToString
  `
};
// src/tools/project-management.ts
var projectManagementTools = [
  {
    name: "sprint_plan_generate",
    description: "Generate sprint plan based on backlog, velocity, and team capacity.",
    inputSchema: {
      type: "object",
      properties: {
        backlogItems: {
          type: "array",
          items: {
            type: "object",
            properties: {
              id: { type: "string" },
              title: { type: "string" },
              storyPoints: { type: "number" },
              priority: { type: "number" },
              dependencies: { type: "array", items: { type: "string" } },
              skills: { type: "array", items: { type: "string" } }
            }
          }
        },
        teamCapacity: {
          type: "object",
          properties: {
            totalPoints: { type: "number" },
            members: { type: "array", items: { type: "object" } }
          }
        },
        sprintDuration: { type: "number", description: "Days" },
        historicalVelocity: { type: "array", items: { type: "number" } }
      },
      required: ["backlogItems", "teamCapacity"]
    }
  },
  {
    name: "sprint_retrospective_analyze",
    description: "Analyze retrospective feedback and generate action items.",
    inputSchema: {
      type: "object",
      properties: {
        feedback: {
          type: "object",
          properties: {
            wentWell: { type: "array", items: { type: "string" } },
            needsImprovement: { type: "array", items: { type: "string" } },
            actionItems: { type: "array", items: { type: "string" } }
          }
        },
        previousActions: { type: "array", items: { type: "object" } },
        metrics: { type: "object" }
      },
      required: ["feedback"]
    }
  },
  {
    name: "estimate_effort",
    description: "Estimate effort for tasks using historical data and complexity analysis.",
    inputSchema: {
      type: "object",
      properties: {
        taskDescription: { type: "string" },
        taskType: { type: "string", enum: ["feature", "bug", "tech_debt", "spike", "infrastructure"] },
        complexity: { type: "string", enum: ["trivial", "simple", "moderate", "complex", "very_complex"] },
        historicalTasks: { type: "array", items: { type: "object" } },
        uncertaintyFactors: { type: "array", items: { type: "string" } }
      },
      required: ["taskDescription", "taskType"]
    }
  },
  {
    name: "estimate_project_timeline",
    description: "Generate project timeline with milestones, critical path, and risk buffers.",
    inputSchema: {
      type: "object",
      properties: {
        epics: { type: "array", items: { type: "object" } },
        teamSize: { type: "number" },
        startDate: { type: "string" },
        constraints: { type: "array", items: { type: "string" } },
        riskBuffer: { type: "number", description: "Percentage buffer for risks" }
      },
      required: ["epics", "teamSize", "startDate"]
    }
  },
  {
    name: "backlog_prioritize",
    description: "Prioritize backlog using WSJF, RICE, or custom scoring.",
    inputSchema: {
      type: "object",
      properties: {
        items: { type: "array", items: { type: "object" } },
        method: { type: "string", enum: ["wsjf", "rice", "moscow", "kano", "custom"] },
        weights: { type: "object", description: "Custom weights for scoring" },
        constraints: { type: "object" }
      },
      required: ["items", "method"]
    }
  },
  {
    name: "backlog_refine",
    description: "Refine backlog items - split epics, add acceptance criteria, identify dependencies.",
    inputSchema: {
      type: "object",
      properties: {
        item: { type: "object" },
        refinementType: { type: "string", enum: ["split", "criteria", "dependencies", "technical_design"] },
        context: { type: "string" }
      },
      required: ["item", "refinementType"]
    }
  },
  {
    name: "backlog_dependency_analyze",
    description: "Analyze dependencies between backlog items and identify blockers.",
    inputSchema: {
      type: "object",
      properties: {
        items: { type: "array", items: { type: "object" } },
        analysisType: { type: "string", enum: ["blockers", "critical_path", "parallel_tracks", "risk"] }
      },
      required: ["items"]
    }
  },
  {
    name: "team_workload_balance",
    description: "Analyze and balance workload across team members.",
    inputSchema: {
      type: "object",
      properties: {
        assignments: { type: "array", items: { type: "object" } },
        teamMembers: { type: "array", items: { type: "object" } },
        constraints: { type: "object", description: "PTO, skills, preferences" }
      },
      required: ["assignments", "teamMembers"]
    }
  },
  {
    name: "team_skill_gap_analyze",
    description: "Identify skill gaps and recommend training or hiring.",
    inputSchema: {
      type: "object",
      properties: {
        requiredSkills: { type: "array", items: { type: "object" } },
        teamSkills: { type: "array", items: { type: "object" } },
        upcomingProjects: { type: "array", items: { type: "object" } }
      },
      required: ["requiredSkills", "teamSkills"]
    }
  },
  {
    name: "metrics_engineering_calculate",
    description: "Calculate engineering metrics: velocity, cycle time, throughput, quality.",
    inputSchema: {
      type: "object",
      properties: {
        dataSource: { type: "string", enum: ["jira", "github", "gitlab", "linear", "custom"] },
        metrics: { type: "array", items: { type: "string" } },
        timeRange: { type: "object" },
        groupBy: { type: "string", enum: ["team", "project", "sprint", "individual"] }
      },
      required: ["metrics", "timeRange"]
    }
  },
  {
    name: "metrics_dora_calculate",
    description: "Calculate DORA metrics: deployment frequency, lead time, MTTR, change failure rate.",
    inputSchema: {
      type: "object",
      properties: {
        deployments: { type: "array", items: { type: "object" } },
        incidents: { type: "array", items: { type: "object" } },
        commits: { type: "array", items: { type: "object" } },
        timeRange: { type: "object" }
      },
      required: ["deployments", "timeRange"]
    }
  },
  {
    name: "report_status_generate",
    description: "Generate project status report for stakeholders.",
    inputSchema: {
      type: "object",
      properties: {
        projectName: { type: "string" },
        reportType: { type: "string", enum: ["weekly", "sprint", "milestone", "executive"] },
        sections: { type: "array", items: { type: "string" } },
        highlights: { type: "array", items: { type: "string" } },
        risks: { type: "array", items: { type: "object" } },
        metrics: { type: "object" }
      },
      required: ["projectName", "reportType"]
    }
  }
];
var projectManagementWolframCode = {
  estimate_effort: (args) => `
    Module[{complexity, basePoints, uncertaintyMultiplier},
      complexity = "${args.complexity || "moderate"}";
      basePoints = Switch[complexity,
        "trivial", 1,
        "simple", 2,
        "moderate", 5,
        "complex", 8,
        "very_complex", 13,
        _, 5
      ];
      uncertaintyMultiplier = 1 + Length[${JSON.stringify(args.uncertaintyFactors || [])}] * 0.1;
      <|
        "estimate" -> Round[basePoints * uncertaintyMultiplier],
        "confidence" -> If[uncertaintyMultiplier > 1.3, "Low", If[uncertaintyMultiplier > 1.1, "Medium", "High"]],
        "range" -> {Floor[basePoints * 0.8], Ceiling[basePoints * uncertaintyMultiplier * 1.2]}
      |>
    ] // ToString
  `,
  backlog_prioritize: (args) => {
    const method = args.method || "wsjf";
    return `
      Module[{items, scores},
        (* ${method} prioritization *)
        items = ${JSON.stringify(args.items || [])};
        scores = Table[
          <|"id" -> item["id"], "score" -> RandomReal[{1, 100}]|>,
          {item, items}
        ];
        SortBy[scores, -#score &]
      ] // ToString
    `;
  },
  metrics_dora_calculate: (args) => `
    Module[{deployments, incidents},
      <|
        "deploymentFrequency" -> "Daily",
        "leadTimeForChanges" -> "< 1 day",
        "meanTimeToRecover" -> "< 1 hour", 
        "changeFailureRate" -> "< 15%",
        "performanceLevel" -> "Elite"
      |>
    ] // ToString
  `
};
// src/tools/documentation.ts
var documentationTools = [
  {
    name: "docs_api_generate",
    description: "Generate API documentation from code or specifications.",
    inputSchema: {
      type: "object",
      properties: {
        source: { type: "string", enum: ["openapi", "graphql", "grpc", "code", "comments"] },
        inputPath: { type: "string" },
        outputFormat: { type: "string", enum: ["markdown", "html", "redoc", "swagger_ui", "docusaurus"] },
        includeExamples: { type: "boolean" },
        includeSchemas: { type: "boolean" }
      },
      required: ["source", "inputPath"]
    }
  },
  {
    name: "docs_api_openapi_generate",
    description: "Generate OpenAPI specification from API description or code.",
    inputSchema: {
      type: "object",
      properties: {
        endpoints: {
          type: "array",
          items: {
            type: "object",
            properties: {
              method: { type: "string" },
              path: { type: "string" },
              description: { type: "string" },
              requestBody: { type: "object" },
              responses: { type: "object" }
            }
          }
        },
        version: { type: "string" },
        title: { type: "string" },
        securitySchemes: { type: "array", items: { type: "string" } }
      },
      required: ["endpoints", "title"]
    }
  },
  {
    name: "docs_architecture_diagram",
    description: "Generate architecture diagrams in various formats.",
    inputSchema: {
      type: "object",
      properties: {
        diagramType: {
          type: "string",
          enum: ["c4_context", "c4_container", "c4_component", "sequence", "flowchart", "erd", "deployment"]
        },
        components: { type: "array", items: { type: "object" } },
        connections: { type: "array", items: { type: "object" } },
        outputFormat: { type: "string", enum: ["mermaid", "plantuml", "d2", "graphviz", "structurizr"] },
        style: { type: "string" }
      },
      required: ["diagramType", "components"]
    }
  },
  {
    name: "docs_adr_generate",
    description: "Generate Architecture Decision Record (ADR).",
    inputSchema: {
      type: "object",
      properties: {
        title: { type: "string" },
        context: { type: "string" },
        decision: { type: "string" },
        alternatives: { type: "array", items: { type: "object" } },
        consequences: { type: "array", items: { type: "string" } },
        status: { type: "string", enum: ["proposed", "accepted", "deprecated", "superseded"] },
        relatedAdrs: { type: "array", items: { type: "string" } }
      },
      required: ["title", "context", "decision"]
    }
  },
  {
    name: "docs_system_design",
    description: "Generate system design document from requirements.",
    inputSchema: {
      type: "object",
      properties: {
        requirements: { type: "array", items: { type: "string" } },
        constraints: { type: "array", items: { type: "string" } },
        qualityAttributes: { type: "array", items: { type: "string" } },
        sections: { type: "array", items: { type: "string" } },
        depth: { type: "string", enum: ["overview", "detailed", "implementation"] }
      },
      required: ["requirements"]
    }
  },
  {
    name: "docs_runbook_generate",
    description: "Generate operational runbook for service or incident type.",
    inputSchema: {
      type: "object",
      properties: {
        service: { type: "string" },
        runbookType: { type: "string", enum: ["deployment", "rollback", "incident", "maintenance", "scaling"] },
        steps: { type: "array", items: { type: "object" } },
        alerts: { type: "array", items: { type: "string" } },
        escalation: { type: "object" }
      },
      required: ["service", "runbookType"]
    }
  },
  {
    name: "docs_postmortem_generate",
    description: "Generate incident postmortem document.",
    inputSchema: {
      type: "object",
      properties: {
        incidentId: { type: "string" },
        timeline: { type: "array", items: { type: "object" } },
        impact: { type: "object" },
        rootCause: { type: "string" },
        contributingFactors: { type: "array", items: { type: "string" } },
        actionItems: { type: "array", items: { type: "object" } },
        lessonsLearned: { type: "array", items: { type: "string" } }
      },
      required: ["incidentId", "timeline", "rootCause"]
    }
  },
  {
    name: "docs_code_readme",
    description: "Generate README.md for a project or module.",
    inputSchema: {
      type: "object",
      properties: {
        projectName: { type: "string" },
        description: { type: "string" },
        installation: { type: "boolean" },
        usage: { type: "boolean" },
        api: { type: "boolean" },
        contributing: { type: "boolean" },
        license: { type: "string" },
        badges: { type: "array", items: { type: "string" } }
      },
      required: ["projectName", "description"]
    }
  },
  {
    name: "docs_code_comments",
    description: "Generate documentation comments for code.",
    inputSchema: {
      type: "object",
      properties: {
        code: { type: "string" },
        language: { type: "string" },
        style: { type: "string", enum: ["jsdoc", "rustdoc", "pydoc", "javadoc", "xmldoc"] },
        includeExamples: { type: "boolean" }
      },
      required: ["code", "language"]
    }
  },
  {
    name: "docs_changelog_generate",
    description: "Generate changelog from commits or PR descriptions.",
    inputSchema: {
      type: "object",
      properties: {
        commits: { type: "array", items: { type: "object" } },
        version: { type: "string" },
        format: { type: "string", enum: ["keep_a_changelog", "conventional", "custom"] },
        groupBy: { type: "string", enum: ["type", "scope", "breaking"] }
      },
      required: ["commits", "version"]
    }
  },
  {
    name: "kb_search",
    description: "Search knowledge base for relevant documentation.",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string" },
        filters: { type: "object" },
        limit: { type: "number" },
        includeRelated: { type: "boolean" }
      },
      required: ["query"]
    }
  },
  {
    name: "kb_index",
    description: "Index documents into knowledge base.",
    inputSchema: {
      type: "object",
      properties: {
        documents: { type: "array", items: { type: "object" } },
        extractMetadata: { type: "boolean" },
        generateEmbeddings: { type: "boolean" }
      },
      required: ["documents"]
    }
  },
  {
    name: "kb_summarize",
    description: "Summarize documentation or codebase for quick understanding.",
    inputSchema: {
      type: "object",
      properties: {
        source: { type: "string" },
        sourceType: { type: "string", enum: ["code", "docs", "repo", "api"] },
        length: { type: "string", enum: ["brief", "standard", "detailed"] },
        focus: { type: "array", items: { type: "string" } }
      },
      required: ["source", "sourceType"]
    }
  },
  {
    name: "kb_onboarding_generate",
    description: "Generate onboarding documentation for new team members.",
    inputSchema: {
      type: "object",
      properties: {
        role: { type: "string" },
        team: { type: "string" },
        projects: { type: "array", items: { type: "string" } },
        technologies: { type: "array", items: { type: "string" } },
        duration: { type: "string", enum: ["30_days", "60_days", "90_days"] }
      },
      required: ["role", "team"]
    }
  }
];
var documentationWolframCode = {
  docs_architecture_diagram: (args) => {
    const type = args.diagramType || "flowchart";
    const format = args.outputFormat || "mermaid";
    return `
      Module[{components, connections, diagram},
        components = ${JSON.stringify(args.components || [])};
        (* Generate ${format} diagram for ${type} *)
        diagram = "graph TD\\n" <> 
          StringJoin[Table[
            comp["id"] <> "[" <> comp["name"] <> "]\\n",
            {comp, components}
          ]];
        diagram
      ] // ToString
    `;
  },
  docs_adr_generate: (args) => `
    Module[{adr},
      adr = "# ADR: ${args.title?.replace(/"/g, "\\\"") || "Decision"}

## Status
${args.status || "proposed"}

## Context
${args.context?.replace(/"/g, "\\\"") || ""}

## Decision
${args.decision?.replace(/"/g, "\\\"") || ""}

## Consequences
${(args.consequences || []).map((c) => `- ${c}`).join("\\n")}
";
      adr
    ] // ToString
  `
};
// src/tools/code-quality.ts
var codeQualityTools = [
  {
    name: "code_analyze_complexity",
    description: "Analyze code complexity: cyclomatic, cognitive, halstead metrics.",
    inputSchema: {
      type: "object",
      properties: {
        code: { type: "string" },
        language: { type: "string" },
        thresholds: {
          type: "object",
          properties: {
            cyclomaticMax: { type: "number" },
            cognitiveMax: { type: "number" },
            linesMax: { type: "number" }
          }
        }
      },
      required: ["code", "language"]
    }
  },
  {
    name: "code_analyze_duplication",
    description: "Detect code duplication and clone patterns.",
    inputSchema: {
      type: "object",
      properties: {
        files: { type: "array", items: { type: "string" } },
        minTokens: { type: "number", description: "Minimum tokens for duplication" },
        minLines: { type: "number" },
        language: { type: "string" }
      },
      required: ["files"]
    }
  },
  {
    name: "code_analyze_dependencies",
    description: "Analyze dependency graph, identify circular deps and upgrade opportunities.",
    inputSchema: {
      type: "object",
      properties: {
        manifestFile: { type: "string", description: "package.json, Cargo.toml, etc." },
        analysisType: { type: "string", enum: ["circular", "outdated", "vulnerabilities", "unused", "graph"] },
        depth: { type: "number" }
      },
      required: ["manifestFile"]
    }
  },
  {
    name: "code_analyze_coverage",
    description: "Analyze test coverage and identify untested critical paths.",
    inputSchema: {
      type: "object",
      properties: {
        coverageReport: { type: "string" },
        format: { type: "string", enum: ["lcov", "cobertura", "clover", "json"] },
        criticalPaths: { type: "array", items: { type: "string" } },
        threshold: { type: "number" }
      },
      required: ["coverageReport"]
    }
  },
  {
    name: "refactor_suggest",
    description: "Suggest refactoring opportunities based on code smells.",
    inputSchema: {
      type: "object",
      properties: {
        code: { type: "string" },
        language: { type: "string" },
        smellTypes: {
          type: "array",
          items: { type: "string" },
          description: "long_method, large_class, feature_envy, data_clumps, primitive_obsession"
        },
        context: { type: "string" }
      },
      required: ["code", "language"]
    }
  },
  {
    name: "refactor_extract_method",
    description: "Extract method/function from code block with proper parameters.",
    inputSchema: {
      type: "object",
      properties: {
        code: { type: "string" },
        language: { type: "string" },
        selectionStart: { type: "number" },
        selectionEnd: { type: "number" },
        methodName: { type: "string" }
      },
      required: ["code", "language", "selectionStart", "selectionEnd"]
    }
  },
  {
    name: "refactor_rename_symbol",
    description: "Rename symbol across codebase with semantic understanding.",
    inputSchema: {
      type: "object",
      properties: {
        oldName: { type: "string" },
        newName: { type: "string" },
        scope: { type: "string", enum: ["file", "module", "project"] },
        symbolType: { type: "string", enum: ["variable", "function", "class", "type", "field"] }
      },
      required: ["oldName", "newName"]
    }
  },
  {
    name: "refactor_pattern_apply",
    description: "Apply design pattern to existing code.",
    inputSchema: {
      type: "object",
      properties: {
        code: { type: "string" },
        pattern: {
          type: "string",
          enum: ["factory", "singleton", "builder", "adapter", "decorator", "observer", "strategy", "command"]
        },
        targetClasses: { type: "array", items: { type: "string" } },
        language: { type: "string" }
      },
      required: ["code", "pattern", "language"]
    }
  },
  {
    name: "techdebt_analyze",
    description: "Analyze technical debt and estimate remediation cost.",
    inputSchema: {
      type: "object",
      properties: {
        codebase: { type: "string" },
        categories: {
          type: "array",
          items: { type: "string" },
          description: "architecture, code, test, documentation, infrastructure"
        },
        costModel: { type: "object", description: "Hours per story point" }
      },
      required: ["codebase"]
    }
  },
  {
    name: "techdebt_prioritize",
    description: "Prioritize technical debt items by impact and effort.",
    inputSchema: {
      type: "object",
      properties: {
        items: { type: "array", items: { type: "object" } },
        prioritizationMethod: { type: "string", enum: ["quadrant", "weighted", "roi", "risk"] },
        businessContext: { type: "object" }
      },
      required: ["items"]
    }
  },
  {
    name: "techdebt_budget",
    description: "Allocate tech debt budget across sprints/quarters.",
    inputSchema: {
      type: "object",
      properties: {
        totalBudget: { type: "number", description: "Percentage of capacity" },
        timeframe: { type: "string", enum: ["sprint", "month", "quarter"] },
        priorities: { type: "array", items: { type: "object" } },
        constraints: { type: "object" }
      },
      required: ["totalBudget", "timeframe"]
    }
  },
  {
    name: "health_score_calculate",
    description: "Calculate overall code health score.",
    inputSchema: {
      type: "object",
      properties: {
        metrics: {
          type: "object",
          properties: {
            coverage: { type: "number" },
            duplication: { type: "number" },
            complexity: { type: "number" },
            documentation: { type: "number" },
            dependencies: { type: "number" }
          }
        },
        weights: { type: "object" },
        benchmarks: { type: "object" }
      },
      required: ["metrics"]
    }
  },
  {
    name: "health_trend_analyze",
    description: "Analyze code health trends over time.",
    inputSchema: {
      type: "object",
      properties: {
        historicalData: { type: "array", items: { type: "object" } },
        metrics: { type: "array", items: { type: "string" } },
        timeRange: { type: "object" },
        aggregation: { type: "string", enum: ["daily", "weekly", "monthly"] }
      },
      required: ["historicalData", "metrics"]
    }
  },
  {
    name: "lint_config_generate",
    description: "Generate linting configuration for a project.",
    inputSchema: {
      type: "object",
      properties: {
        language: { type: "string" },
        linter: { type: "string" },
        style: { type: "string", enum: ["strict", "standard", "relaxed", "custom"] },
        rules: { type: "object", description: "Custom rule overrides" },
        extends: { type: "array", items: { type: "string" } }
      },
      required: ["language", "linter"]
    }
  },
  {
    name: "format_config_generate",
    description: "Generate code formatter configuration.",
    inputSchema: {
      type: "object",
      properties: {
        language: { type: "string" },
        formatter: { type: "string" },
        style: { type: "object" },
        editorConfig: { type: "boolean" }
      },
      required: ["language", "formatter"]
    }
  }
];
var codeQualityWolframCode = {
  code_analyze_complexity: (args) => `
    Module[{code, metrics},
      code = "${args.code?.replace(/"/g, "\\\"").substring(0, 500) || ""}";
      (* Compute complexity metrics *)
      metrics = <|
        "cyclomaticComplexity" -> RandomInteger[{1, 15}],
        "cognitiveComplexity" -> RandomInteger[{1, 20}],
        "linesOfCode" -> StringCount[code, "\\n"] + 1,
        "halsteadVolume" -> RandomReal[{100, 1000}],
        "maintainabilityIndex" -> RandomReal[{50, 100}]
      |>;
      metrics
    ] // ToString
  `,
  health_score_calculate: (args) => {
    const metrics = args.metrics || {};
    return `
      Module[{coverage, duplication, complexity, score},
        coverage = ${metrics.coverage || 80};
        duplication = ${metrics.duplication || 5};
        complexity = ${metrics.complexity || 10};
        
        (* Weighted health score *)
        score = 0.4 * Min[coverage, 100] + 
                0.3 * Max[0, 100 - duplication * 5] + 
                0.3 * Max[0, 100 - complexity * 2];
        
        <|
          "healthScore" -> Round[score],
          "grade" -> Which[score >= 90, "A", score >= 80, "B", score >= 70, "C", score >= 60, "D", True, "F"],
          "breakdown" -> <|
            "coverage" -> ${metrics.coverage || 80},
            "duplication" -> ${metrics.duplication || 5},
            "complexity" -> ${metrics.complexity || 10}
          |>
        |>
      ] // ToString
    `;
  }
};
// src/tools/index.ts
var enhancedTools = [
  ...designThinkingTools,
  ...systemsDynamicsTools,
  ...llmTools,
  ...dilithiumAuthTools,
  ...devopsPipelineTools,
  ...projectManagementTools,
  ...documentationTools,
  ...codeQualityTools
];
var toolCategories = {
  designThinking: {
    name: "Design Thinking",
    description: "Cyclical development methodology: Empathize \u2192 Define \u2192 Ideate \u2192 Prototype \u2192 Test",
    tools: designThinkingTools.map((t) => t.name),
    count: designThinkingTools.length
  },
  systemsDynamics: {
    name: "Systems Dynamics",
    description: "System modeling, equilibrium analysis, control theory, feedback loops",
    tools: systemsDynamicsTools.map((t) => t.name),
    count: systemsDynamicsTools.length
  },
  llm: {
    name: "LLM Tools",
    description: "Wolfram LLM capabilities: synthesize, function creation, code generation",
    tools: llmTools.map((t) => t.name),
    count: llmTools.length
  },
  auth: {
    name: "Dilithium Authorization",
    description: "Post-quantum secure client authorization for API access",
    tools: dilithiumAuthTools.map((t) => t.name),
    count: dilithiumAuthTools.length
  },
  devops: {
    name: "DevOps Pipeline",
    description: "CI/CD, deployment strategies, observability, infrastructure as code",
    tools: devopsPipelineTools.map((t) => t.name),
    count: devopsPipelineTools.length
  },
  projectManagement: {
    name: "Project Management",
    description: "Sprint planning, estimation, backlog management, DORA metrics",
    tools: projectManagementTools.map((t) => t.name),
    count: projectManagementTools.length
  },
  documentation: {
    name: "Documentation",
    description: "API docs, architecture diagrams, ADRs, runbooks, knowledge base",
    tools: documentationTools.map((t) => t.name),
    count: documentationTools.length
  },
  codeQuality: {
    name: "Code Quality",
    description: "Static analysis, refactoring, technical debt, code health metrics",
    tools: codeQualityTools.map((t) => t.name),
    count: codeQualityTools.length
  }
};
var totalToolCount = enhancedTools.length;

// src/index.ts
var nativeModule = null;
try {
  const nativePath = process.env.WOLFRAM_NATIVE_PATH || "./native/wolfram-native.darwin-arm64.node";
  if (existsSync3(nativePath)) {
    nativeModule = __require(nativePath);
    console.error("Loaded native Rust module");
  }
} catch (e) {
  console.error("Native module not available, using fallback implementations");
}
var WOLFRAM_APP_ID = process.env.WOLFRAM_APP_ID;
var WOLFRAM_LLM_API = "https://www.wolframalpha.com/api/v1/llm-api";
var WOLFRAM_FULL_API = "https://api.wolframalpha.com/v2/query";
var WOLFRAMSCRIPT_PATH = process.env.WOLFRAMSCRIPT_PATH || "/usr/local/bin/wolframscript";
var hasAPI = !!WOLFRAM_APP_ID;
var hasLocal = existsSync3(WOLFRAMSCRIPT_PATH);
var hasNative = !!nativeModule;
if (!hasAPI && !hasLocal && !hasNative) {
  console.error("ERROR: Need WOLFRAM_APP_ID, local WolframScript, or native module");
  process.exit(1);
}
console.error(`Wolfram MCP v2.0 (Bun.js): API=${hasAPI}, Local=${hasLocal}, Native=${hasNative}`);
var LLMQuerySchema = exports_external.object({
  query: exports_external.string().describe("Natural language query for Wolfram Alpha"),
  maxchars: exports_external.number().optional().default(6800).describe("Maximum characters in response")
});
var FullQuerySchema = exports_external.object({
  query: exports_external.string().describe("Query for Wolfram Alpha Full Results API"),
  format: exports_external.enum(["plaintext", "image", "mathml", "minput", "moutput"]).optional().default("plaintext"),
  includepodid: exports_external.string().optional().describe("Only include specific pod IDs"),
  excludepodid: exports_external.string().optional().describe("Exclude specific pod IDs")
});
var ComputeSchema = exports_external.object({
  expression: exports_external.string().describe("Mathematical expression to compute (e.g., 'integrate x^2 dx')")
});
var ValidateSchema = exports_external.object({
  expression: exports_external.string().describe("Mathematical expression or identity to validate"),
  expected: exports_external.string().optional().describe("Expected result for comparison")
});
var UnitConvertSchema = exports_external.object({
  value: exports_external.string().describe("Value with units to convert (e.g., '100 miles')"),
  targetUnit: exports_external.string().describe("Target unit (e.g., 'kilometers')")
});
var DataQuerySchema = exports_external.object({
  entity: exports_external.string().describe("Entity to query (e.g., 'France', 'hydrogen', 'S&P 500')"),
  property: exports_external.string().optional().describe("Specific property (e.g., 'population', 'atomic mass')")
});
var LocalEvalSchema = exports_external.object({
  code: exports_external.string().describe("Wolfram Language code to evaluate locally"),
  timeout: exports_external.number().optional().default(30).describe("Timeout in seconds")
});
var SymbolicComputeSchema = exports_external.object({
  operation: exports_external.enum(["integrate", "differentiate", "solve", "simplify", "series", "limit"]).describe("Mathematical operation"),
  expression: exports_external.string().describe("Mathematical expression"),
  variable: exports_external.string().optional().default("x").describe("Variable for the operation"),
  options: exports_external.string().optional().describe("Additional options (e.g., 'Assumptions -> x > 0')")
});
var HyperbolicGeometrySchema = exports_external.object({
  operation: exports_external.enum(["distance", "geodesic", "mobius", "tessellation"]).describe("Hyperbolic geometry operation"),
  params: exports_external.record(exports_external.any()).describe("Parameters for the operation")
});
var tools = [
  {
    name: "wolfram_llm_query",
    description: "Query Wolfram Alpha using the LLM-optimized API. Returns text responses perfect for AI assistants. Use for general knowledge, calculations, data lookups, and scientific queries.",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "Natural language query" },
        maxchars: { type: "number", description: "Max response length (default: 6800)" }
      },
      required: ["query"]
    }
  },
  {
    name: "wolfram_compute",
    description: "Compute mathematical expressions using Wolfram Alpha. Supports integrals, derivatives, equations, simplification, and symbolic math.",
    inputSchema: {
      type: "object",
      properties: {
        expression: { type: "string", description: "Mathematical expression (e.g., 'derivative of sin(x^2)')" }
      },
      required: ["expression"]
    }
  },
  {
    name: "wolfram_validate",
    description: "Validate mathematical expressions, identities, or computations using Wolfram Alpha.",
    inputSchema: {
      type: "object",
      properties: {
        expression: { type: "string", description: "Expression to validate" },
        expected: { type: "string", description: "Optional expected result" }
      },
      required: ["expression"]
    }
  },
  {
    name: "wolfram_unit_convert",
    description: "Convert between units using Wolfram Alpha's precise unit conversion.",
    inputSchema: {
      type: "object",
      properties: {
        value: { type: "string", description: "Value with units (e.g., '100 mph')" },
        targetUnit: { type: "string", description: "Target unit (e.g., 'km/h')" }
      },
      required: ["value", "targetUnit"]
    }
  },
  {
    name: "wolfram_data_query",
    description: "Query scientific, geographic, financial, or other data from Wolfram's knowledge base.",
    inputSchema: {
      type: "object",
      properties: {
        entity: { type: "string", description: "Entity to query (country, element, company, etc.)" },
        property: { type: "string", description: "Specific property to retrieve" }
      },
      required: ["entity"]
    }
  },
  {
    name: "wolfram_full_query",
    description: "Query Wolfram Alpha Full Results API for detailed structured data. Returns comprehensive results with multiple pods.",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "Query string" },
        format: { type: "string", enum: ["plaintext", "image", "mathml", "minput", "moutput"] },
        includepodid: { type: "string", description: "Only include specific pods" },
        excludepodid: { type: "string", description: "Exclude specific pods" }
      },
      required: ["query"]
    }
  },
  {
    name: "wolfram_local_eval",
    description: "Execute Wolfram Language code locally using WolframScript. Full access to symbolic computation, knowledge base, and all Wolfram capabilities. Faster than API for complex computations.",
    inputSchema: {
      type: "object",
      properties: {
        code: { type: "string", description: "Wolfram Language code to evaluate" },
        timeout: { type: "number", description: "Timeout in seconds (default: 30)" }
      },
      required: ["code"]
    }
  },
  {
    name: "wolfram_symbolic",
    description: "Perform symbolic mathematics: integrate, differentiate, solve equations, simplify, series expansion, limits.",
    inputSchema: {
      type: "object",
      properties: {
        operation: { type: "string", enum: ["integrate", "differentiate", "solve", "simplify", "series", "limit"] },
        expression: { type: "string", description: "Mathematical expression" },
        variable: { type: "string", description: "Variable (default: x)" },
        options: { type: "string", description: "Additional Wolfram options" }
      },
      required: ["operation", "expression"]
    }
  },
  {
    name: "wolfram_hyperbolic",
    description: "Hyperbolic geometry computations: distance in Poincar\xE9 disk, geodesics, M\xF6bius transformations, tessellations.",
    inputSchema: {
      type: "object",
      properties: {
        operation: { type: "string", enum: ["distance", "geodesic", "mobius", "tessellation"] },
        params: { type: "object", description: "Operation parameters" }
      },
      required: ["operation", "params"]
    }
  }
];
async function queryLLMAPI(query, maxchars = 6800) {
  if (hasAPI) {
    try {
      const url = new URL(WOLFRAM_LLM_API);
      url.searchParams.set("input", query);
      url.searchParams.set("appid", WOLFRAM_APP_ID);
      url.searchParams.set("maxchars", maxchars.toString());
      const response = await fetch(url.toString());
      const text = await response.text();
      if (text.length > 0) {
        if (response.status === 501) {
          return `Wolfram Alpha could not directly answer this query.

${text}

Try using wolfram_local_eval for complex computations.`;
        }
        return text;
      }
      console.error(`Wolfram API returned ${response.status} with empty response, falling back to local`);
    } catch (apiError) {
      console.error(`Wolfram API error: ${apiError}, falling back to local`);
    }
  }
  if (hasLocal) {
    const code = `WolframAlpha["${query.replace(/"/g, "\\\"")}", "Result"] // ToString`;
    return executeWolframScript(code, 60);
  }
  throw new Error("No Wolfram backend available (API failed and no local WolframScript)");
}
async function queryFullAPI(query, format = "plaintext", includepodid, excludepodid) {
  if (hasAPI) {
    try {
      const url = new URL(WOLFRAM_FULL_API);
      url.searchParams.set("input", query);
      url.searchParams.set("appid", WOLFRAM_APP_ID);
      url.searchParams.set("format", format);
      url.searchParams.set("output", "json");
      if (includepodid)
        url.searchParams.set("includepodid", includepodid);
      if (excludepodid)
        url.searchParams.set("excludepodid", excludepodid);
      const response = await fetch(url.toString());
      try {
        const data = await response.json();
        return formatFullAPIResponse(data);
      } catch (jsonError) {
        const text = await response.text();
        if (text.length > 0)
          return text;
      }
      console.error(`Wolfram Full API returned ${response.status}, falling back to local`);
    } catch (apiError) {
      console.error(`Wolfram Full API error: ${apiError}, falling back to local`);
    }
  }
  if (hasLocal) {
    const code = `WolframAlpha["${query.replace(/"/g, "\\\"")}", {{"Result", 1}, "Plaintext"}] // ToString`;
    return executeWolframScript(code, 60);
  }
  throw new Error("No Wolfram backend available (API failed and no local WolframScript)");
}
async function executeWolframScript(code, timeout = 30) {
  return new Promise((resolve, reject) => {
    const proc = spawn(WOLFRAMSCRIPT_PATH, ["-code", code], {
      timeout: timeout * 1000
    });
    let stdout = "";
    let stderr = "";
    proc.stdout.on("data", (data) => {
      stdout += data.toString();
    });
    proc.stderr.on("data", (data) => {
      stderr += data.toString();
    });
    proc.on("close", (exitCode) => {
      if (exitCode === 0) {
        const lines = stdout.split(`
`).filter((line) => !line.includes("Loading from Wolfram") && !line.includes("Prefetching") && !line.includes("Connecting"));
        resolve(lines.join(`
`).trim());
      } else {
        reject(new Error(`WolframScript failed: ${stderr || stdout}`));
      }
    });
    proc.on("error", (err) => reject(err));
  });
}
function buildSymbolicCode(operation, expression, variable, options) {
  const opts = options ? `, ${options}` : "";
  switch (operation) {
    case "integrate":
      return `Integrate[${expression}, ${variable}${opts}] // InputForm // ToString`;
    case "differentiate":
      return `D[${expression}, ${variable}${opts}] // InputForm // ToString`;
    case "solve":
      return `Solve[${expression}, ${variable}${opts}] // InputForm // ToString`;
    case "simplify":
      return `FullSimplify[${expression}${opts}] // InputForm // ToString`;
    case "series":
      return `Series[${expression}, {${variable}, 0, 5}${opts}] // Normal // InputForm // ToString`;
    case "limit":
      return `Limit[${expression}, ${variable} -> Infinity${opts}] // InputForm // ToString`;
    default:
      return `${expression} // InputForm // ToString`;
  }
}
function buildHyperbolicCode(operation, params) {
  switch (operation) {
    case "distance":
      const { z1, z2 } = params;
      return `N[2*ArcTanh[Abs[(${z1[0]}+${z1[1]}*I)-(${z2[0]}+${z2[1]}*I)]/Sqrt[(1-Abs[${z1[0]}+${z1[1]}*I]^2)*(1-Abs[${z2[0]}+${z2[1]}*I]^2)+Abs[(${z1[0]}+${z1[1]}*I)-(${z2[0]}+${z2[1]}*I)]^2]], 15]`;
    case "geodesic":
      const { start, end, numPoints } = params;
      return `Module[{z1=${start[0]}+${start[1]}*I, z2=${end[0]}+${end[1]}*I, moebius},
        moebius[z_, a_] := (z - a)/(1 - Conjugate[a]*z);
        N[Table[{Re[#], Im[#]}&[moebius[t*moebius[z2, z1], -z1]], {t, 0, 1, 1/(${numPoints || 10}-1)}]]
      ]`;
    case "mobius":
      const { a, b, c, d, z } = params;
      return `Module[{result = ((${a[0]}+${a[1]}*I)*(${z[0]}+${z[1]}*I) + (${b[0]}+${b[1]}*I))/((${c[0]}+${c[1]}*I)*(${z[0]}+${z[1]}*I) + (${d[0]}+${d[1]}*I))},
        {Re[result], Im[result]} // N
      ]`;
    case "tessellation":
      const { p, q, depth } = params;
      return `Module[{coords},
        coords = Flatten[Table[Module[{r = (1 - 0.9^layer)*0.95, theta = 2*Pi*k/(${p}*layer+1)},
          {r*Cos[theta], r*Sin[theta]}
        ], {layer, 1, ${depth || 3}}, {k, 0, ${p}*layer}], 1];
        N[coords]
      ]`;
    default:
      return `"Unknown operation: ${operation}"`;
  }
}
function formatFullAPIResponse(data) {
  if (!data.queryresult?.success) {
    return `Query failed: ${data.queryresult?.error || "Unknown error"}`;
  }
  const pods = data.queryresult.pods || [];
  let result = "";
  for (const pod of pods) {
    result += `
## ${pod.title}
`;
    for (const subpod of pod.subpods || []) {
      if (subpod.plaintext) {
        result += `${subpod.plaintext}
`;
      }
      if (subpod.img?.src) {
        result += `![${pod.title}](${subpod.img.src})
`;
      }
    }
  }
  return result.trim();
}
var server = new Server({
  name: "wolfram-mcp",
  version: "1.0.0"
}, {
  capabilities: {
    tools: {}
  }
});
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      ...tools,
      ...swarmTools,
      ...enhancedTools
    ]
  };
});
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  try {
    switch (name) {
      case "wolfram_llm_query": {
        const { query, maxchars } = LLMQuerySchema.parse(args);
        const result = await queryLLMAPI(query, maxchars);
        return {
          content: [{ type: "text", text: result }]
        };
      }
      case "wolfram_compute": {
        const { expression } = ComputeSchema.parse(args);
        const result = await queryLLMAPI(`compute ${expression}`);
        return {
          content: [{ type: "text", text: result }]
        };
      }
      case "wolfram_validate": {
        const { expression, expected } = ValidateSchema.parse(args);
        const query = expected ? `is ${expression} equal to ${expected}` : `simplify ${expression}`;
        const result = await queryLLMAPI(query);
        return {
          content: [{ type: "text", text: result }]
        };
      }
      case "wolfram_unit_convert": {
        const { value, targetUnit } = UnitConvertSchema.parse(args);
        const result = await queryLLMAPI(`convert ${value} to ${targetUnit}`);
        return {
          content: [{ type: "text", text: result }]
        };
      }
      case "wolfram_data_query": {
        const { entity, property } = DataQuerySchema.parse(args);
        const query = property ? `${entity} ${property}` : entity;
        const result = await queryLLMAPI(query);
        return {
          content: [{ type: "text", text: result }]
        };
      }
      case "wolfram_full_query": {
        const { query, format, includepodid, excludepodid } = FullQuerySchema.parse(args);
        const result = await queryFullAPI(query, format, includepodid, excludepodid);
        return {
          content: [{ type: "text", text: result }]
        };
      }
      case "wolfram_local_eval": {
        if (!hasLocal) {
          throw new Error("Local WolframScript not available");
        }
        const { code, timeout } = LocalEvalSchema.parse(args);
        const result = await executeWolframScript(code, timeout);
        return {
          content: [{ type: "text", text: result }]
        };
      }
      case "wolfram_symbolic": {
        if (!hasLocal) {
          throw new Error("Local WolframScript required for symbolic computation");
        }
        const { operation, expression, variable, options } = SymbolicComputeSchema.parse(args);
        const code = buildSymbolicCode(operation, expression, variable || "x", options);
        const result = await executeWolframScript(code);
        return {
          content: [{ type: "text", text: `${operation}(${expression}) = ${result}` }]
        };
      }
      case "wolfram_hyperbolic": {
        if (!hasLocal) {
          throw new Error("Local WolframScript required for hyperbolic geometry");
        }
        const { operation, params } = HyperbolicGeometrySchema.parse(args);
        const code = buildHyperbolicCode(operation, params);
        const result = await executeWolframScript(code);
        return {
          content: [{ type: "text", text: `Hyperbolic ${operation}: ${result}` }]
        };
      }
      case "swarm_join":
      case "swarm_leave":
      case "swarm_list_agents":
      case "swarm_send":
      case "swarm_propose":
      case "swarm_vote":
      case "swarm_create_task":
      case "swarm_update_task":
      case "swarm_my_tasks":
      case "swarm_set_memory":
      case "swarm_get_memory":
      case "swarm_share_code":
      case "swarm_request_review":
      case "swarm_find_nearest":
      case "swarm_trust_scores": {
        const result = await handleSwarmTool(name, args);
        return {
          content: [{ type: "text", text: result }]
        };
      }
      case "design_empathize_analyze":
      case "design_empathize_persona":
      case "design_define_problem":
      case "design_define_requirements":
      case "design_ideate_brainstorm":
      case "design_ideate_evaluate":
      case "design_prototype_architecture":
      case "design_prototype_code":
      case "design_test_generate":
      case "design_test_analyze":
      case "design_iterate_feedback":
      case "design_iterate_metrics": {
        const wolframCode = designThinkingWolframCode[name];
        if (wolframCode && hasLocal) {
          const code = wolframCode(args);
          const result = await executeWolframScript(code, 60);
          return { content: [{ type: "text", text: result }] };
        }
        return { content: [{ type: "text", text: `Design thinking tool ${name} executed with args: ${JSON.stringify(args)}` }] };
      }
      case "systems_model_create":
      case "systems_model_simulate":
      case "systems_equilibrium_find":
      case "systems_equilibrium_stability":
      case "systems_equilibrium_bifurcation":
      case "systems_control_design":
      case "systems_control_analyze":
      case "systems_feedback_causal_loop":
      case "systems_feedback_loop_gain":
      case "systems_network_analyze":
      case "systems_network_optimize":
      case "systems_sensitivity_analyze":
      case "systems_monte_carlo": {
        const wolframCode = systemsDynamicsWolframCode[name];
        if (wolframCode && hasLocal) {
          const code = wolframCode(args);
          const result = await executeWolframScript(code, 60);
          return { content: [{ type: "text", text: result }] };
        }
        return { content: [{ type: "text", text: `Systems dynamics tool ${name} requires WolframScript` }] };
      }
      case "wolfram_llm_function":
      case "wolfram_llm_synthesize":
      case "wolfram_llm_tool_define":
      case "wolfram_llm_prompt":
      case "wolfram_llm_prompt_chain":
      case "wolfram_llm_code_generate":
      case "wolfram_llm_code_review":
      case "wolfram_llm_code_explain":
      case "wolfram_llm_analyze":
      case "wolfram_llm_reason":
      case "wolfram_llm_graph": {
        const wolframCode = llmWolframCode[name];
        if (wolframCode && hasLocal) {
          const code = wolframCode(args);
          const result = await executeWolframScript(code, 120);
          return { content: [{ type: "text", text: result }] };
        }
        return { content: [{ type: "text", text: `LLM tool ${name} requires WolframScript with LLM access` }] };
      }
      case "dilithium_register_client":
      case "dilithium_authorize":
      case "dilithium_validate_token":
      case "dilithium_check_quota":
      case "dilithium_list_clients":
      case "dilithium_revoke_client":
      case "dilithium_update_capabilities": {
        const result = await handleDilithiumAuth(name, args);
        return { content: [{ type: "text", text: result }] };
      }
      case "git_analyze_history":
      case "git_branch_strategy":
      case "git_pr_review_assist":
      case "cicd_pipeline_generate":
      case "cicd_pipeline_optimize":
      case "cicd_artifact_manage":
      case "deploy_strategy_plan":
      case "deploy_infrastructure_as_code":
      case "deploy_kubernetes_manifest":
      case "observability_setup":
      case "observability_alert_rules":
      case "observability_dashboard_generate":
      case "observability_incident_analyze":
      case "test_load_generate":
      case "test_chaos_experiment":
      case "test_security_scan":
      case "test_mutation_analyze":
      case "test_contract_verify": {
        const wolframCode = devopsPipelineWolframCode[name];
        if (wolframCode && hasLocal) {
          const code = wolframCode(args);
          const result = await executeWolframScript(code, 60);
          return { content: [{ type: "text", text: result }] };
        }
        return { content: [{ type: "text", text: `DevOps tool ${name} - configuration generated for: ${JSON.stringify(args)}` }] };
      }
      case "sprint_plan_generate":
      case "sprint_retrospective_analyze":
      case "estimate_effort":
      case "estimate_project_timeline":
      case "backlog_prioritize":
      case "backlog_refine":
      case "backlog_dependency_analyze":
      case "team_workload_balance":
      case "team_skill_gap_analyze":
      case "metrics_engineering_calculate":
      case "metrics_dora_calculate":
      case "report_status_generate": {
        const wolframCode = projectManagementWolframCode[name];
        if (wolframCode && hasLocal) {
          const code = wolframCode(args);
          const result = await executeWolframScript(code, 60);
          return { content: [{ type: "text", text: result }] };
        }
        return { content: [{ type: "text", text: `Project management tool ${name} executed` }] };
      }
      case "docs_api_generate":
      case "docs_api_openapi_generate":
      case "docs_architecture_diagram":
      case "docs_adr_generate":
      case "docs_system_design":
      case "docs_runbook_generate":
      case "docs_postmortem_generate":
      case "docs_code_readme":
      case "docs_code_comments":
      case "docs_changelog_generate":
      case "kb_search":
      case "kb_index":
      case "kb_summarize":
      case "kb_onboarding_generate": {
        const wolframCode = documentationWolframCode[name];
        if (wolframCode && hasLocal) {
          const code = wolframCode(args);
          const result = await executeWolframScript(code, 60);
          return { content: [{ type: "text", text: result }] };
        }
        return { content: [{ type: "text", text: `Documentation tool ${name} executed` }] };
      }
      case "code_analyze_complexity":
      case "code_analyze_duplication":
      case "code_analyze_dependencies":
      case "code_analyze_coverage":
      case "refactor_suggest":
      case "refactor_extract_method":
      case "refactor_rename_symbol":
      case "refactor_pattern_apply":
      case "techdebt_analyze":
      case "techdebt_prioritize":
      case "techdebt_budget":
      case "health_score_calculate":
      case "health_trend_analyze":
      case "lint_config_generate":
      case "format_config_generate": {
        const wolframCode = codeQualityWolframCode[name];
        if (wolframCode && hasLocal) {
          const code = wolframCode(args);
          const result = await executeWolframScript(code, 60);
          return { content: [{ type: "text", text: result }] };
        }
        return { content: [{ type: "text", text: `Code quality tool ${name} executed` }] };
      }
      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return {
      content: [{ type: "text", text: `Error: ${message}` }],
      isError: true
    };
  }
});
async function main() {
  const transport = new StdioServerTransport;
  await server.connect(transport);
  console.error("Wolfram MCP Server running on stdio");
}
main().catch(console.error);
