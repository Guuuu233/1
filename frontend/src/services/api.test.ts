import { describe, expect, it } from 'vitest'

import { ApiError, isNotFoundError } from '@/services/api'

describe('ApiError', () => {
    it('carries the HTTP status alongside the message', () => {
        const error = new ApiError('job not found', 404)
        expect(error).toBeInstanceOf(Error)
        expect(error.message).toBe('job not found')
        expect(error.status).toBe(404)
        expect(error.name).toBe('ApiError')
    })
})

describe('isNotFoundError', () => {
    it('matches ApiError with status 404', () => {
        expect(isNotFoundError(new ApiError('job not found', 404))).toBe(true)
        expect(isNotFoundError(new ApiError('报告不存在', 404))).toBe(true)
    })

    it('rejects other statuses and plain errors', () => {
        expect(isNotFoundError(new ApiError('Internal error', 500))).toBe(false)
        expect(isNotFoundError(new ApiError('Unauthorized', 401))).toBe(false)
        expect(isNotFoundError(new Error('job not found'))).toBe(false)
        expect(isNotFoundError(null)).toBe(false)
        expect(isNotFoundError('job not found')).toBe(false)
    })
})
