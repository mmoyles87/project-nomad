/**
 * Tests for the embedding-model decision — which model the knowledge base is
 * embedded and searched with, from the `rag.embeddingModel` setting.
 *
 * The decision is a pure function; RagService does only the KV read around it.
 *
 * Pure functions only — no MySQL, Redis, Qdrant, or Ollama needed:
 *   npm run test:unit
 */
import * as assert from 'node:assert/strict'
import { test } from 'node:test'

import { EMBEDDING_MODEL_NAME } from '../../constants/ollama.js'
import { isEmbeddingModelName, pickEmbeddingModel } from '../../app/utils/misc.js'

test('embedding model: unset setting keeps nomic-embed-text and its prefixes', () => {
  for (const configured of [null, undefined, '', '   ']) {
    assert.deepEqual(pickEmbeddingModel(configured), {
      name: EMBEDDING_MODEL_NAME,
      dimension: 768,
      documentPrefix: 'search_document: ',
      queryPrefix: 'search_query: ',
    })
  }
})

test('embedding model: bge-m3 is 1024-wide and takes no prefixes', () => {
  assert.deepEqual(pickEmbeddingModel('bge-m3'), {
    name: 'bge-m3',
    dimension: 1024,
    documentPrefix: '',
    queryPrefix: '',
  })
})

test('embedding model: surrounding whitespace is trimmed before matching', () => {
  assert.equal(pickEmbeddingModel('  bge-m3  ').name, 'bge-m3')
})

test('embedding model: an unknown model falls back to the default', () => {
  // A value written before a model was removed from EMBEDDING_MODELS, or by hand.
  // Its dimension and prefixes are unknown, so it cannot be embedded with safely.
  assert.equal(pickEmbeddingModel('mxbai-embed-large').name, EMBEDDING_MODEL_NAME)
})

test('embedding model: object prototype keys are not models', () => {
  // `'toString' in {}` is true; a lookup that used `in` would return a
  // profile with no dimension and create a collection of size undefined.
  assert.equal(pickEmbeddingModel('toString').name, EMBEDDING_MODEL_NAME)
  assert.equal(pickEmbeddingModel('constructor').name, EMBEDDING_MODEL_NAME)
})

test('embedding model names: bge-m3 is recognised with or without the :latest tag', () => {
  // Ollama lists a bare `ollama pull bge-m3` as bge-m3:latest, and the name has
  // no "embed" in it, so the chat model lists need this to leave it out.
  assert.equal(isEmbeddingModelName('bge-m3'), true)
  assert.equal(isEmbeddingModelName('bge-m3:latest'), true)
  assert.equal(isEmbeddingModelName(EMBEDDING_MODEL_NAME), true)
})

test('embedding model names: chat models and prototype keys are not embedding models', () => {
  assert.equal(isEmbeddingModelName('llama3.1:8b'), false)
  assert.equal(isEmbeddingModelName('toString'), false)
})
