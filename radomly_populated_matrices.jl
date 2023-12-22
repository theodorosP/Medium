#this function can create a sparce matrix radomly populated
#percentage = the population percentage
#matrix is the sparse matrix we want to populate

function randomly_populate_matrix(percentage, matrix)
  seed = 1
  row_index_list = Vector{Int64}()
  col_index_list = Vector{Int64}()
  row_aux = Vector{Int64}()
    for i in range(1, percentage* size(matrix, 1)* size(matrix, 2))
      seed += 1 
      Random.seed!(seed)
      row_index =  sample(1:size(matrix, 1), 1, replace = false)
      col_index =  sample(1:size(matrix, 2), 1, replace = false)
      append!(row_index_list, row_index)
      append!(col_index_list, col_index)
    end
    for i in range(1, length(row_index_list))
      if row_index_list[i] == col_index_list[i]
        if row_index_list[i] < size(matrix, 1)
          row_index_list[i] += 1
        else
          row_index_list[i] -= 1 
        end
      end
    end
  #zipped = sortperm(collect(zip(row_index_list, col_index_list)))
  zipped = tuple.(row_index_list, col_index_list)    
  l = unique(zipped)
  #up_range = length(zipped) - 1, so that I will get excactly the number of zipped elements
  for i in range(length(l), length(zipped) - 1)
    seed += 1
      Random.seed!(seed)
      row_index =  rand(1: size(matrix, 1))    
      col_index =  rand(1: size(matrix, 2))
      while row_index == col_index
        row_index =  rand(1: size(matrix, 1))    
        col_index =  rand(1: size(matrix, 2))
      end
      k = (row_index, col_index)
      while k in l 
        seed += 1
        Random.seed!(seed)
        row_index =  rand(1: size(matrix, 1))    
        col_index =  rand(1: size(matrix, 2))
        while row_index == col_index
          row_index =  rand(1: size(matrix, 1))    
          col_index =  rand(1: size(matrix, 2))
        end
        k = (row_index, col_index)
    end
 push!(l, k)
end
r = Vector{Int64}()
c = Vector{Int64}()
for i in l 
  append!(r, i[1])
  append!(c, i[2])
end
for i in range(1, length(r))
  matrix[r[i], c[i]] = 1
end
return matrix
end    
