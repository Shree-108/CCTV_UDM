'use client'

import { useMemo, useState, type ReactNode } from 'react'
import { ArrowDown, ArrowUp, ChevronLeft, ChevronRight, Columns3 } from 'lucide-react'
import { cx } from '../../lib/format'
import { IconButton } from './primitives'

export interface Column<T> {
  key: string
  header: ReactNode
  cell: (row: T) => ReactNode
  sortValue?: (row: T) => number | string | null
  align?: 'left' | 'right' | 'center'
  width?: string
  hideable?: boolean
  defaultHidden?: boolean
  className?: string
}

/**
 * Compact operational table: client-side sort, column visibility, optional row
 * selection, keyboard-activatable rows, and either client or server pagination.
 */
export function DataTable<T>({ rows, columns, rowKey, onRowClick, selectedKey, pageSize = 25, serverPage, empty, dense, selectable, selected, onSelectedChange, toolbar }: {
  rows: T[]
  columns: Column<T>[]
  rowKey: (r: T) => string | number
  onRowClick?: (r: T) => void
  selectedKey?: string | number | null
  pageSize?: number
  serverPage?: { offset: number; total: number; limit: number; onChange: (offset: number) => void }
  empty?: ReactNode
  dense?: boolean
  selectable?: boolean
  selected?: Set<string | number>
  onSelectedChange?: (s: Set<string | number>) => void
  toolbar?: ReactNode
}) {
  const [sort, setSort] = useState<{ key: string; dir: 1 | -1 } | null>(null)
  const [page, setPage] = useState(0)
  const [hidden, setHidden] = useState<Set<string>>(() => new Set(columns.filter((c) => c.defaultHidden).map((c) => c.key)))
  const [colMenu, setColMenu] = useState(false)
  const visible = columns.filter((c) => !hidden.has(c.key))

  const sorted = useMemo(() => {
    if (!sort) return rows
    const col = columns.find((c) => c.key === sort.key)
    if (!col?.sortValue) return rows
    return [...rows].sort((a, b) => {
      const va = col.sortValue!(a), vb = col.sortValue!(b)
      if (va === vb) return 0
      if (va === null) return 1
      if (vb === null) return -1
      return (va < vb ? -1 : 1) * sort.dir
    })
  }, [rows, sort, columns])

  const pageRows = serverPage ? sorted : sorted.slice(page * pageSize, (page + 1) * pageSize)
  const total = serverPage ? serverPage.total : rows.length
  const offset = serverPage ? serverPage.offset : page * pageSize
  const limit = serverPage ? serverPage.limit : pageSize
  const hideable = columns.filter((c) => c.hideable)

  return (
    <div className="flex min-w-0 flex-col">
      {(toolbar || hideable.length > 0) && (
        <div className="flex items-center justify-between gap-2 border-b border-line px-3 py-2">
          <div className="flex min-w-0 flex-wrap items-center gap-2">{toolbar}</div>
          {hideable.length > 0 && (
            <div className="relative">
              <IconButton icon={Columns3} label="Choose columns" onClick={() => setColMenu((v) => !v)} active={colMenu} />
              {colMenu && (
                <div className="absolute right-0 z-20 mt-1 w-44 rounded-md border border-line bg-elev p-1.5 shadow-[var(--shadow)]">
                  {hideable.map((c) => (
                    <label key={c.key} className="flex cursor-pointer items-center gap-2 rounded px-1.5 py-1 text-[12px] text-fg-2 hover:bg-hover">
                      <input type="checkbox" checked={!hidden.has(c.key)} onChange={() => setHidden((h) => {
                        const n = new Set(h)
                        if (n.has(c.key)) n.delete(c.key)
                        else n.add(c.key)
                        return n
                      })} />
                      {c.header}
                    </label>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}
      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-[12.5px]">
          <thead>
            <tr className="border-b border-line bg-panel-2/60 text-left">
              {selectable && (
                <th className="w-8 px-3">
                  <input type="checkbox" aria-label="Select all on page"
                    checked={pageRows.length > 0 && pageRows.every((r) => selected?.has(rowKey(r)))}
                    onChange={(e) => {
                      const n = new Set(selected)
                      pageRows.forEach((r) => (e.target.checked ? n.add(rowKey(r)) : n.delete(rowKey(r))))
                      onSelectedChange?.(n)
                    }} />
                </th>
              )}
              {visible.map((c) => (
                <th key={c.key} scope="col" style={{ width: c.width }}
                  aria-sort={sort?.key === c.key ? (sort.dir === 1 ? 'ascending' : 'descending') : undefined}
                  className={cx('whitespace-nowrap px-3 py-2 text-[11px] font-medium uppercase tracking-wide text-fg-3',
                    c.align === 'right' && 'text-right', c.align === 'center' && 'text-center')}>
                  {c.sortValue ? (
                    <button className="inline-flex items-center gap-1 uppercase hover:text-fg"
                      onClick={() => setSort((s) => (s?.key === c.key ? (s.dir === 1 ? { key: c.key, dir: -1 } : null) : { key: c.key, dir: 1 }))}>
                      {c.header}
                      {sort?.key === c.key && (sort.dir === 1 ? <ArrowUp className="size-3" /> : <ArrowDown className="size-3" />)}
                    </button>
                  ) : c.header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {pageRows.length === 0 && (
              <tr><td colSpan={visible.length + (selectable ? 1 : 0)}>{empty}</td></tr>
            )}
            {pageRows.map((r) => {
              const k = rowKey(r)
              return (
                <tr key={k}
                  tabIndex={onRowClick ? 0 : undefined}
                  onClick={() => onRowClick?.(r)}
                  onKeyDown={(e) => { if (onRowClick && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); onRowClick(r) } }}
                  className={cx('border-b border-line/70 transition-colors', onRowClick && 'cursor-pointer hover:bg-hover focus:bg-hover focus:outline-none',
                    selectedKey === k && 'bg-info-bg')}>
                  {selectable && (
                    <td className="px-3" onClick={(e) => e.stopPropagation()}>
                      <input type="checkbox" aria-label="Select row" checked={!!selected?.has(k)}
                        onChange={() => {
                          const n = new Set(selected)
                          if (n.has(k)) n.delete(k)
                          else n.add(k)
                          onSelectedChange?.(n)
                        }} />
                    </td>
                  )}
                  {visible.map((c) => (
                    <td key={c.key} className={cx('px-3 text-fg-2', dense ? 'py-1' : 'py-1.5', c.align === 'right' && 'text-right tabular',
                      c.align === 'center' && 'text-center', c.className)}>{c.cell(r)}</td>
                  ))}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      {total > limit && (
        <div className="flex items-center justify-between border-t border-line px-3 py-1.5 text-[12px] text-fg-3">
          <span className="tabular">{offset + 1}–{Math.min(offset + limit, total)} of {total.toLocaleString()}</span>
          <div className="flex gap-1">
            <IconButton icon={ChevronLeft} label="Previous page" disabled={offset === 0}
              onClick={() => (serverPage ? serverPage.onChange(Math.max(0, offset - limit)) : setPage((p) => p - 1))} />
            <IconButton icon={ChevronRight} label="Next page" disabled={offset + limit >= total}
              onClick={() => (serverPage ? serverPage.onChange(offset + limit) : setPage((p) => p + 1))} />
          </div>
        </div>
      )}
    </div>
  )
}
